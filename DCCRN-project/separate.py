#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import os
import pickle
import librosa
import torch
from torch.utils.data import DataLoader
import wav_loader_c as loader
import train_utils
import numpy
import re
import model as model
import time
#from data import EvalDataLoader, EvalDataset
#from conv_tasnet_change import ConvTasNet
#from utils import remove_pad
inner_print = print 

parser = argparse.ArgumentParser('Separate speech using Conv-TasNet')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model file created by training')
parser.add_argument('--mix_dir', type=str, default=None,
                    help='Directory including mixture wav files')
parser.add_argument('--mix_json', type=str, default=None,
                    help='Json file including mixture wav files')
parser.add_argument('--out_dir', type=str, default='exp/result',
                    help='Directory putting separated wav files')
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU to separate speech')
parser.add_argument('--sample_rate', default=16000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')
#def print(*arg):
   #inner_#print(*arg)  
   #inner_#print(*arg,file=open("log.txt","a")) 
   
   
   #inner_print(time.strftime("%T"),"\t",*arg,file=open("log.txt","a")) 
def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - numpy.mean(ref_sig)
    out_sig = out_sig - numpy.mean(out_sig)
    ref_energy = numpy.sum(ref_sig ** 2) + eps
    proj = numpy.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = numpy.sum(proj ** 2) / (numpy.sum(noise ** 2) + eps)
    sisnr = 10 * numpy.log(ratio + eps) / numpy.log(10.0)
    return sisnr
   


def separate(args):
    if args.mix_dir is None and args.mix_json is None:
        print("Must provide mix_dir or mix_json! When providing mix_dir, "
              "mix_json is ignored.")

    # Load model
    '''
    model = ConvTasNet.load_model(args.model_path)
    #print(model)
    model.eval()
    '''
    model_dict = model.DCCRN().load_model('/root/Kind_Of_Net/DCCRN/DCCRN-Wu/logs/final.pth.tar') #/root/DCCRN-master_killed/logs/final.pth.tar
    #Model = model.DCCRN()
    
    print("start")
    model_dict.eval()
    
    dns_home = "/root/text/"  # dir of dns-datas  
    save_file = "./logs"  # model save
########################################################################

    batch_size = 160  # calculate batch_size
    load_batch = 4  # load batch_size(not calculate)
    device = torch.device("cuda:0")  # device

    lr = 0.001  # learning_rate
    # load train and test name , train:test=4:1
    def get_all_names(train_test, dns_home):
        
        test_names = train_test["test"]

        
        test_noisy_names = []
        test_clean_names = []

        
        for name in test_names:
            code = str(name).split('_')[-1]
            clean_file = os.path.join(dns_home, 'clean', '%s' % code)
            noisy_file = os.path.join(dns_home, 'noisy', name)
            test_clean_names.append(clean_file)
            test_noisy_names.append(noisy_file)
        return test_noisy_names, test_clean_names
    def get_train_test_name(dns_home):
        all_name = []
        for i in os.walk(os.path.join(dns_home, "noisy")):
            for name in i[2]:
                all_name.append(name)
        
        test_names = all_name
        
        #print(len(test_names))
        data = {"test": test_names}
        pickle.dump(data, open("./train_test2_names.data", "wb"))
        return data
    train_test = get_train_test_name(dns_home)
    test_noisy_names, test_clean_names = \
        get_all_names(train_test, dns_home=dns_home)
    list_shape = numpy.array(test_noisy_names).shape
    print(list_shape)

    test_dataset = loader.WavDataset(test_noisy_names, test_clean_names, frame_dur=37.5)
    # dataloader
    
    test_dataloader = loader.AudioDataLoader(test_dataset, batch_size=load_batch, shuffle=True)
    # Load data
    print(111)
    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    initials = []
    Mix = []
    def write(inumpyuts, filename, sr=args.sample_rate):
        ##inumpyuts= inumpyuts - numpy.mean(inumpyuts)     
        ##inumpyuts = inumpyuts/numpy.max(numpy.abs(inumpyuts)) 
        librosa.output.write_wav(filename, inumpyuts, sr)# norm=True)
    with torch.no_grad():
        loss_sum = 0
        i = 0
        for i, (data) in enumerate(test_dataloader):
            x, y, test_clean_names = data
            #print(111)
                ##x = x.view(x.size(0) * x.size(1), x.size(2))
                ##y = y.view(y.size(0) * y.size(1), y.size(2))
                ##shuffle = torch.randperm(x.size(0))
                ##x = x[shuffle]
                ##y = y[shuffle]
                ###print(padded_mixture.size())
                
                ##print(x.size())# 10,1,96000
            y=y.squeeze(1)
            
            
            start = time.time()
            #x = x[:1,:,:600]
            #print("x",x.size())
            estimate_source = model_dict(x) ##breakpoint 
            estimate_source = model_dict(estimate_source)
            #print("time: ",time.time() - start)
            #print(estimate_source.size())

            results = numpy.array(estimate_source)
            results = results.reshape(load_batch,-1)
            Mix = numpy.array(x)
            Mix = Mix.reshape(load_batch,-1)
            initials = numpy.array(y)
            initials = initials.reshape(load_batch,-1)

            ##loss_sum += cal_SISNR(initials,results)## - cal_SISNR(initials,Mix)
            for i, filename in enumerate(test_clean_names):
                print(filename[0])
                nums = re.sub("\D","",filename[0])
                loss_sum += cal_SISNR(initials[i-1],results[i-1]) - cal_SISNR(initials[i-1],Mix[i-1])
                outpath1 = '/root/Kind_Of_Net/DCCRN/DCCRN-Wu/exp/result/' + str(nums) +'.wav'
                outpath2 = '/root/Kind_Of_Net/DCCRN/DCCRN-Wu/exp/result/' + str(nums) +'_deal.wav'
                outpath3 = '/root/Kind_Of_Net/DCCRN/DCCRN-Wu/exp/result/' + str(nums) +'_mix.wav'

                write(initials[i-1],outpath1,sr = 16000)
                write(results[i-1],outpath2,sr = 16000)
                write(Mix[i-1],outpath3,sr = 16000)
        print(loss_sum/30)

    


if __name__ == '__main__':
    args = parser.parse_args()
    #print(args)
    separate(args)

