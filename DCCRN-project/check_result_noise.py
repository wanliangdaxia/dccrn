import argparse
import os

from collections import OrderedDict
import librosa
from mir_eval.separation import bss_eval_sources
import numpy as np
import torch
from torch.utils.data import DataLoader
import wav_loader_c as loader
#from pit_criterion import cal_losscon
from model  import DCCRN
#from utils import remove_pad
#from pypesq import pesq
import time
import librosa
import soundfile as sf

inner_print = print 
log_name = 'dccrn_output'
def print(*arg):
   '''
   '''
   inner_print(time.strftime("%T"),"\t",*arg,file=open(log_name+".txt","a")) 
   '''
   '''
parser = argparse.ArgumentParser('Evaluate separation performance using model_rnn')
#放置训练好的模型
parser.add_argument('--model_path', type=str, default='/root/wanliang/DCCRN-project/howl_DCCRN_project/epoch43.pth.tar', 
                    help='Path to model file created by training')
parser.add_argument('--data_dir', type=str, default='/root/Dual-Path-Transformer-Network-PyTorch-main/data/tt',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')
def evaluate(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model = DCCRN()
    
    model.cuda()
    model_info = torch.load(args.model_path)
    state_dict = OrderedDict()
    for k, v in model_info['model_state_dict'].items():
        name = k.replace("module.", "")    # remove 'module.'
        state_dict[name] = v
    model.load_state_dict(state_dict)
    print(model)
    #加载test数据
    with open('/root/rain/data/test_noisy.txt', 'r') as test_file_noisy:
        test_noisy_names = [line.strip() for line in test_file_noisy.readlines()]
    with open('/root/howl/crn/test-clean-dccrn.txt', 'r') as test_file_clean:
        test_clean_names = [line.strip() for line in test_file_clean.readlines()]

    test_dataset = loader.WavDataset(test_noisy_names, test_clean_names, frame_dur=37.5)
    ts_loader = loader.AudioDataLoader(test_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        i = 0
        total =0
        pesq=0
        for i, (data) in enumerate(ts_loader):
            x, y, test_clean_names = data

            if args.use_cuda:
                x = x.cuda()
                #y = y.cuda()
                
            estimate_source = model(x)
       
            
            #直接减
            x = x.reshape(-1).cpu() - estimate_source.reshape(-1).cpu()
            
            #自适应滤波器
            # estimate_source = estimate_source.reshape(-1).cpu()
            # x = x.reshape(-1).cpu()
            # estimate_source = np.expand_dims(estimate_source,axis=0)
            # x = np.expand_dims(x,axis=0)

            # x,Cw =LMS(estimate_source,x,T=160000, L=16, mu=0.001)



            #输出模型处理后的数据
            sf.write('/root/wanliang/DCCRN-project/output/{}.wav'.format(i),x,samplerate=16000,format='WAV', subtype='PCM_16')


def LMS(x,lpb,T, L, mu):
    x=x
    lpb=lpb
	# Initiate the system
    Cx = np.zeros((1,L)) 		# The state of C(z)
    Cw = np.zeros((1,L)) 		# The weight of C(z)
    e_cont = np.zeros((1,T)) 		# Data buffer for the control error
    Xhx = np.zeros((1,L))			# The state of the filtered x(k)
    y = np.zeros((1,T))
    #x.shape=[1,160000]
    #lpb.shape=[1,160000]
    lamda = 1e-6
	#And apply the FxLMS algorithm
    for k in range(0,T):#T=160000
        Cx = np.roll(Cx,1)
        
        # Cx[0,0] = x[0,k]
        # y[0,k] = np.dot(Cx,Cw[0,:])
        Cx[0,0] = x[0,k]
        y[0,k] = np.dot(Cx,Cw[0,:])
  
        e_cont[0,k] = lpb[0,k] - y[0,k]
        Xhx = np.roll(Xhx,1)
        Xhx[0,0] = x[0,k]
        Cw = Cw + mu*e_cont[0,k]*Xhx/(np.dot(Xhx,Xhx.T)+lamda)

    return e_cont[0],Cw[0]



if __name__ == '__main__':
    args = parser.parse_args()

    evaluate(args)