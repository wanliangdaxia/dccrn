# coding: utf-8
# AuthoangTianRui
# Date 2020/11/3 16:49
# from base.BaseModel import *
import torch
import torch.nn as nn

from utils.conv_stft import *
from utils.complexnn import *
from scipy import fftpack
import numpy
import time


class DCCRN(nn.Module):
    def __init__(self,
                 rnn_layer=3, 
                 rnn_hidden=256,
                 win_len=400, 
                 hop_len=100, 
                 fft_len=512, 
                 win_type='hanning',
                 use_clstm=True, 
                 use_cbn=True, 
                #  use_clstm=False, 
                #  use_cbn=False, 
                 masking_mode='E',
                 kernel_size=5, 
                 kernel_num=(32,64,128,128,256,256)#use_clstm=True
                 ):
        super(DCCRN, self).__init__()
        self.rnn_layer = rnn_layer                  # 3
        self.rnn_hidden = rnn_hidden                # 256
        self.win_len = win_len                      # 400
        self.hop_len = hop_len                      # 100
        self.fft_len = fft_len                      # 512
        self.win_type = win_type                    # hanning
        self.use_clstm = use_clstm                  # 是否用complex LSTM
        self.use_cbn = use_cbn                      # 是否用Complex BatchNorm
        self.masking_mode = masking_mode            # 此处用E
        self.kernel_size = kernel_size              # 5
        self.kernel_num = (2,) + kernel_num         # (2, 32, 64, 128, 128, 256, 256)

        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, self.win_type, 'complex', fix=True)      # win_len = 400，hop_len = 100，fft_len = 512，win_type = hanning
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, self.win_type, 'complex', fix=True)    # win_len = 400，hop_len = 100，fft_len = 512，win_type = hanning

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        ##################################### Encoder ########################################
        for idx in range(len(self.kernel_num) - 1):                                                             # 0 -> 6
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d(
                        
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),#2代表要看两帧的信息，如果多看前一帧的信息就是因果的，多看后一帧就是非因果的
                        stride=(2, 1),#2代表在高两侧的幅度为2，1代表在宽两侧的幅度为1
                        padding=(2, 1)#2代表在高两侧的幅度为2，1代表在宽两侧的幅度为1
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                        self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )
            
        ##################################### enhance########################################
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))                                              # 512 / 2**7 = 4
        if self.use_clstm:                                                                                      # 此处use_clstm=False,所以不执行
            rnns = []
            for idx in range(rnn_layer):#rnn的层数为3                                                            # 0 -> 3
                rnns.append(
                    NavieComplexLSTM(
                        input_size=hidden_dim * self.kernel_num[-1] if idx == 0 else self.rnn_hidden,           # 如果idx = 0,则input_size = 4 * 512 = 2048。如果idx ！= 0,则input_size = 256。
                        hidden_size=self.rnn_hidden,
                        batch_first=False,
                        projection_dim=hidden_dim * self.kernel_num[-1] if idx == rnn_layer - 1 else None
                    )
                )
                self.enhance = nn.Sequential(*rnns)
        else:                                                                                                   # 此处执行
            self.enhance = nn.LSTM(
                input_size=hidden_dim * self.kernel_num[-1],                                                    # input_size = 4 * 512 = 2048
                hidden_size=self.rnn_hidden,                                                                    # hidden_size = 256
                num_layers=2,
                dropout=0.0,
                batch_first=False
            )            
            self.transform = nn.Linear(self.rnn_hidden, hidden_dim * self.kernel_num[-1])                       # in_features:256 ，out_features:2048
            
        ##################################### decoder ########################################    
        for idx in range(len(self.kernel_num) - 1, 0, -1):                                                      # rang(6, 0, -1) 表示 6 -> 1
            if idx != 1:                                                                                        # decoder 的前5个 有BatchNorm2d 和 PReLU
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],                                                           
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                            self.kernel_num[idx - 1]),
                        nn.PReLU()
                    )
                )
            else:                                                                                               # decoder 的第六个没有BatchNorm2d 和 PReLU
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)

                        )
                    )
                )
        if isinstance(self.enhance, nn.LSTM):                               # 判断是否用了complex LSTM ,如果没有
            self.enhance.flatten_parameters()                               # 为了提高内存的利用率和效率，调用flatten_parameters让parameter的数据存放成contiguous chunk(连续的块)。类似我们调用tensor.contiguous

    def forward(self, x):                                                   # x.shape: ([6, 1, 16000])，因为每个音频的长度都为6s，
        stft = self.stft(x) 
        #print(x.shape)                                                # stft.shape: ([6, 514, 163])
        real = stft[:, :self.fft_len // 2 + 1]                              # 取从stft出来的结果前257项  real.shape: ([6, 257, 163])
        imag = stft[:, self.fft_len // 2 + 1:]                              # 取从stft出来的结果后257项  imag.shape: ([6, 257, 163])
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)                # STFT之后的模
        spec_phase = torch.atan2(imag, real)                                # 实部和虚部的反正切值
        spec_complex = torch.stack([real, imag], dim=1)[:, :, 1:]           # 第一个点不要 实部与虚部拼接  spec_complex:([6, 2, 256, 163])  B,2,256,-1  
        #print(spec_complex.shape)
        out = spec_complex                                                  # 每个 encoder 的输出
        encoder_out = []
        ################################### encoder ######################################
        for idx, encoder in enumerate(self.encoder):                        # 把六个encoder拆开
            out = encoder(out)                                              # 每个 encoder 的输出
            #print("1",out.shape)
            encoder_out.append(out)                                         # 六个encoder 的输出
            #assert 1==2
        #s = numpy.array(encoder_out)
        B, C, D, T = out.size()                                             # 6 512 4 163
        out = out.permute(3, 0, 1, 2)                                       # [163, 6, 512, 4]
        if self.use_clstm:                                                  # 此处不执行
            r_rnn_in = out[:, :, :C // 2]
            i_rnn_in = out[:, :, C // 2:]
            r_rnn_in = torch.reshape(r_rnn_in, [T, B, C // 2 * D])
            i_rnn_in = torch.reshape(i_rnn_in, [T, B, C // 2 * D])
            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])
            r_rnn_in = torch.reshape(r_rnn_in, [T, B, C // 2, D])
            i_rnn_in = torch.reshape(i_rnn_in, [T, B, C // 2, D])
            out = torch.cat([r_rnn_in, i_rnn_in], 2)
        else:                                                               # 此处执行
            out = torch.reshape(out, [T, B, C * D])
            ################################## enhance #####################################
            out, _ = self.enhance(out)
            out = self.transform(out)
            out = torch.reshape(out, [T, B, C, D])        
        out = out.permute(1, 2, 3, 0)                                       # out:([6, 512, 4, 163])
        
        ################################### decoder ######################################
        #print(out.shape)
        for idx in range(len(self.decoder)):         
            out = complex_cat([out, encoder_out[-1 - idx]], 1)            
            out = self.decoder[idx](out)
            #print(out.shape)
            out = out[..., 1:]                                              # 最后一个维度的第一个点不要
            #print("2",out.shape)
        mask_real = out[:, 0]                                               # [6, 256, 163]         out: ([6, 2, 256, 163])
        mask_imag = out[:, 1]                                               # [6, 256, 163]

        mask_real = F.pad(mask_real, [0, 0, 1, 0])                          # [6, 256, 163] -> [6, 257, 163] 第二个维度填充第一个点，值为0
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0])
        if self.masking_mode == 'E':                                        # 此处执行这个
            mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5            # 预测出来的幅度掩码
            real_phase = mask_real / (mask_mags + 1e-8)
            imag_phase = mask_imag / (mask_mags + 1e-8)
            mask_phase = torch.atan2(
                imag_phase,
                real_phase
            )
            mask_mags = torch.tanh(mask_mags)
            est_mags = mask_mags * spec_mags
            est_phase = spec_phase + mask_phase
            real = est_mags * torch.cos(est_phase)
            imag = est_mags * torch.sin(est_phase)
        elif self.masking_mode == 'C':                                          # 此处不执行
            real = real * mask_real - imag * mask_imag
            imag = real * mask_imag + imag * mask_real
        elif self.masking_mode == 'R':                                          # 此处不执行
            real = real * mask_real
            imag = imag * mask_imag

        out_spec = torch.cat([real, imag], 1)
        out_wav = self.istft(out_spec)
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = out_wav.clamp_(-1, 1)
        return out_wav

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls()
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package

def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10(target_norm / (noise_norm + eps) + eps)
    return torch.mean(snr)


def loss(inputs, label):
    return -(si_snr(inputs, label))


if __name__ == '__main__':
    test_model = DCCRN()
    a = torch.randn([1,16000])
    out = test_model(a)
