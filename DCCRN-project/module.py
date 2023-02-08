# coding: utf-8
# AuthorangTianRui
# Date 2020/9/30 10:55


from complex_progress import *
import torchaudio_contrib as audio_nn
from utils import *
from memory_profiler import profile

class ConvSTFT(nn.Module):
    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):      # win_len = 400，win_inc = 100，fft_len = 512，win_type = hanning, feature_type = complex
        super(ConvSTFT, self).__init__()
        if fft_len == None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))                                               # ceil 向上取整
        else:                                                                                                   # 本模型应该取
            self.fft_len = fft_len                                                                              # fft_len = 512

        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)                                      # 加窗
        self.weight = nn.Parameter(kernel, requires_grad=(not fix)) ########################################################################################
        #self.register_buffer('weight', kernel)                                                                  # 向 kernel 中添加一个名为 weight 的buffer。
        self.feature_type = feature_type                                                                        # complex
        self.stride = win_inc                                                                                   # 100
        self.win_len = win_len                                                                                  # 400
        self.dim = self.fft_len                                                                                 # 512

    def forward(self, inputs):                                                                                  # inputs.shape: ([6, 1, 16600])
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        inputs = F.pad(inputs, [self.win_len - self.stride, self.win_len - self.stride])                        # 在inputs 的左边填充300列，右边填充300列。在inputs的最后一个维度+300。 inputs.shape: ([6, 1, 16600])
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)                                             # stride = 100      outputs.shape: ([6, 514, 163])
        if self.feature_type == 'complex':                                                                      # 此处执行这个，直接返回
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real ** 2 + imag ** 2)
            phase = torch.atan2(imag, real)
            return mags, phase


class ConviSTFT(nn.Module):
    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):    # win_len = 400，win_inc = 100，fft_len = 512，win_type = hanning, feature_type = complex
        super(ConviSTFT, self).__init__()
        if fft_len == None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        self.weight = nn.Parameter(kernel, requires_grad=(not fix)) ########################################################################################
        #self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:, None, :])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """
        if phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)
            inputs = torch.cat([real, imag], 1)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)
        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs / (coff + 1e-8)
        # outputs = torch.where(coff == 0, outputs, outputs/coff)
        outputs = outputs[..., self.win_len - self.stride:-(self.win_len - self.stride)]
        return outputs





class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, chw, padding=None):
        super().__init__()
        if padding is None:
            padding = [int((i - 1) / 2) for i in kernel_size]  # same
            # padding
        self.conv = ComplexConv2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                                  stride=stride, padding=padding)
        self.bn = ComplexBatchNormal(chw[0], chw[1], chw[2])
        self.prelu = nn.PReLU()

    def forward(self, x, train):
        x = self.conv(x)
        print(x.shape)
        x = self.bn(x, train)
        x = self.prelu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, chw, padding=None):
        super().__init__()
        self.transconv = ComplexConvTranspose2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                                                stride=stride, padding=padding)
        self.bn = ComplexBatchNormal(chw[0], chw[1], chw[2])
        self.prelu = nn.PReLU()

    def forward(self, x, train=True):
        x = self.transconv(x)
        x = self.bn(x, train)
        x = self.prelu(x)
        return x


class DCCRN(nn.Module):
    
    def __init__(self, net_params, device, batch_size=36):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.encoders = []
        self.lstms = []
        self.dense = ComplexDense(net_params["dense"][0], net_params["dense"][1])
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for index in range(len(en_channels) - 1):
            model = Encoder(
                in_channel=en_channels[index], out_channel=en_channels[index + 1],
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index]
            )
            self.add_module("encoder{%d}" % index, model)
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        for index in range(len(net_params["lstm_dim"]) - 1):
            model = ComplexLSTM(input_size=lstm_dims[index], hidden_size=lstm_dims[index + 1],
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)
            self.add_module("lstm{%d}" % index, model)
        # init decoder
        de_channels = net_params["decoder_channels"]
        de_ker_size = net_params["decoder_kernel_sizes"]
        de_strides = net_params["decoder_strides"]
        de_padding = net_params["decoder_paddings"]
        for index in range(len(de_channels) - 1):
            model = Decoder(
                in_channel=de_channels[index] + en_channels[len(self.encoders) - index],
                out_channel=de_channels[index + 1],
                kernel_size=de_ker_size[index], stride=de_strides[index], padding=de_padding[index],
                chw=decoder_chw[index]
            )
            self.add_module("decoder{%d}" % index, model)
            self.decoders.append(model)

        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)
        self.decoders = nn.ModuleList(self.decoders)
        self.linear = ComplexConv2d(in_channel=2, out_channel=1, kernel_size=1, stride=1)

    def forward(self, x, train=True):
        skiper = []
        for index, encoder in enumerate(self.encoders):
            skiper.append(x)
            x = encoder(x, train)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D)
        lstm_ = lstm_.permute(2, 0, 1, 3)
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_)
        lstm_ = lstm_.permute(1, 0, 2, 3)
        lstm_out = lstm_.reshape(B * T, -1, D)
        dense_out = self.dense(lstm_out)
        dense_out = dense_out.reshape(B, T, C, F, D)
        p = dense_out.permute(0, 2, 3, 1, 4)
        for index, decoder in enumerate(self.decoders):
            p = decoder(p, train)
            p = torch.cat([p, skiper[len(skiper) - index - 1]], dim=1)
        mask = torch.tanh(self.linear(p))
        return mask


class DCCRN_(nn.Module):
    def __init__(self, win_len=512, hop_len=128, fft_len=512, net_params, batch_size=36, device, win_length):
        super().__init__()

        self.stft = ConvSTFT(512, 128, 512, self.win_type,'real', fix=True)
        self.DCCRN = DCCRN(net_params, device=device, batch_size=batch_size)
        self.istft = ConviSTFT(512,128,512,  self.win_type, 'complex', fix=None)

    def forward(self, signal, train=True):
        stft = self.stft(signal)
        mask_predict = self.DCCRN(stft, train=train)
        predict = stft * mask_predict
        clean = self.istft(predict)
        return clean

if __name__ == '__main__':
    
    test_model =DCCRN_()
    #t = np.random.randn(16000 * 4) * 0.001
    #t = np.clip(t, -1, 1)
    input = torch.randn([1,16000])
    #input = torch.from_numpy(t[None, None, :].astype(np.float32))
    w = test_model(input)