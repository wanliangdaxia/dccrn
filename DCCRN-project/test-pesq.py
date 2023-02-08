import argparse
import os

from collections import OrderedDict
import librosa
from mir_eval.separation import bss_eval_sources
import numpy as np
import torch
from torch.utils.data import DataLoader
import wav_loader_c as loader
#from pit_criterion import cal_loss
from model import DCCRN
#from utils import remove_pad
from pypesq import pesq
import time
inner_print = print 
log_name = 'howl_pesq_test'
def print(*arg):
   #inner_print(*arg)  
   #inner_print(*arg,file=open("log.txt","a")) 
   '''
   
   '''
   inner_print(time.strftime("%T"),"\t",*arg,file=open(log_name+".txt","a")) 
   '''
   
   '''
parser = argparse.ArgumentParser('Evaluate separation performance using model_rnn')
parser.add_argument('--model_path', type=str, default='/root/DCCRN-Wu/howl_DCCRN_newway/epoch18.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--data_dir', type=str, default='/root/Dual-Path-Transformer-Network-PyTorch-main/data/tt',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--cal_sdr', type=int, default=1,
                    help='Whether calculate SDR, add this option because calculation of SDR is very slow')
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')

def evaluate(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = DCCRN()
    model.cuda()
    model_info = torch.load(args.model_path)
    state_dict = OrderedDict()
    for k, v in model_info['model_state_dict'].items():
        name = k.replace("module.", "")    # remove 'module.'
        state_dict[name] = v
    model.load_state_dict(state_dict)
    print(model)

    with open('/root/howl/crn/test-noisy-dccrn.txt', 'r') as test_file_noisy:
        test_noisy_names = [line.strip() for line in test_file_noisy.readlines()]
    with open('/root/DCCRN-Wu/enchanced/enchenced.txt', 'r') as test_file_clean:
        test_clean_names = [line.strip() for line in test_file_clean.readlines()]

    test_dataset = loader.WavDataset(test_noisy_names, test_clean_names, frame_dur=37.5)
    ts_loader = loader.AudioDataLoader(test_dataset, batch_size=1, shuffle=True)
    with torch.no_grad():
        total =0
        i = 0
        pesq=0
        for i, (data) in enumerate(ts_loader):
            x, y, test_clean_names = data
            #print("X:x",x.shape)
            if args.use_cuda:
                x = x.cuda()
                y = y.cuda()
            estimate_source = model(x)
            x=estimate_source.reshape(-1).cpu()
            y=y.reshape(-1).cpu()

            score = get_pesq(y, x,sr=16000)

            pesq +=score

            total +=1

            print(pesq)

        avg_pesq = pesq/total

        print('PESQ: {:.4f}'.format(avg_pesq))

def get_pesq(ref, deg, sr):

    score = pesq(ref, deg,sr)

    return score

if __name__ == '__main__':
    args = parser.parse_args()
    
    evaluate(args)