
import librosa
import soundfile as sf
from pypesq import pesq
from pystoi import stoi
from mir_eval.separation import bss_eval_sources
import numpy as np
import os
import re
import torch
#in_dir_c = '/root/Conv-TasNet/egsj0/exp/train_r8000_N256_L16_B256_H512_P3_X8_R2_C1_gLN_causal0_relu_epoch200_half1_norm5_bs6_worker4_adam_lr1e-4_mmt0_l20_trparate/'
#in_dir_c='/root/NABDCRN/DCCRN-master_killed/exp/DCCRN/L5dB/'
in_dir_c='/root/wanliang/DCCRN-New/enchanced_targ_noise_new/'#降噪后的音频地址
in_dir_s1 ='/root/howl/crn/test/clean/' #原本干净语音地址

#in_dir_s2 = '/root/Conv-TasNet/egsj0/data/s2/'
in_dir_mix = '/root/howl/crn/test/noisy/'#noisy地址
file_infos = []
#in_dir_c = os.path.abspath(in_dir_c) #杩斿洖缁濆璺緞
#in_dir_s = os.path.abspath(in_dir_s)
wav_list_c = os.listdir(in_dir_c) #降噪后的音频
#wav_list_s = os.listdir(in_dir_s)#杩斿洖鎸囧畾璺緞涓嬬殑鏂囦欢鍜屾枃浠跺す鍒楄〃 

k=0
cnt =0
#sum=0
sum_denoised = 0
sum_noisy = 0
sum_stoi1 = 0
sum_stoi2 = 0
sum_sdr = 0
sum_n_sdr = 0


def SNR(x1, x2):
    from numpy.linalg import norm
    return 20 * np.log10(norm(x1) / norm(x2))
for wav_file_c in wav_list_c:
    if not wav_file_c.endswith('.wav'):   
        continue
    #shuzi =re.findall(r"\d+\.?\d*",wav_file_c)
    shuzi = re.sub("\D","",wav_file_c)
    
    print("shuzi: ",shuzi)
    #assert 1 == 2
    #if not shuzi[0].endswith('.'):
        
        #shuzi = (re.sub("\D", "", shuzi[0]))
        #print(z)
    indir_s1 = in_dir_s1 + str(shuzi) +'.wav'   #clean signal

    ##indir_c2 = in_dir_s2 + str(shuzi[0]) +'.wav'   
    #indir_c = in_dir_c + str(shuzi) +'_deal.wav'   #denoised signal
    indir_c = in_dir_c + str(shuzi) +'.wav'   #denoised signal
    #indir_s2 = in_dir_c + str(shuzi) +'_s2.wav'
    indir_mix = in_dir_mix + str(shuzi) +'.wav'   #noisy
    y_s, sr = librosa.load(indir_s1,mono=True, sr=None) 
   # print('y_s:',y_s.shape)
    #y_c2, sr = librosa.load(indir_c2,mono=None, sr=None) 
    y_e, sr = librosa.load(indir_c,mono=True, sr=None) 
    #y_s2, sr = librosa.load(indir_s2,mono=None, sr=None) 
    y_mix,sr = librosa.load(indir_mix,mono=True, sr=None) 
    #print("y_mix:",y_mix.shape)
    # y_s = torch.from_numpy(y_s).cuda()
    # y_e = torch.from_numpy(y_e).cuda()
    # y_mix = torch.from_numpy(y_mix).cuda()

    cnt+=1

    #DB = SNR(y_c1,y_mix)
    #sum+=DB

    #print(k)
    #print(sum/k)
    #clean 归一化
    #array = np.array(y_c1)
    #space = np.argmax(array)
    #y_c1 = y_c1/y_c1[space]
    
    
    #denoised 归一化
    #array = np.array(y_s1)
    #space = np.argmax(array)
    #y_s1 = y_s1/y_s1[space]
    
    #array = np.array(y_mix)
    #space = np.argmax(array)
    #y_mix = y_mix/y_mix[space]

    #y_c1 =y_c1.any()
    #print(y_c1.shape)
    score1 = pesq(y_s, y_e, sr)
    score1_stoi = stoi(y_s, y_e, sr)
    #sdr, sir, sar, popt = bss_eval_sources(y_s, y_e)
    print("denoised: ",score1," stoi: ",score1_stoi)#," sdr: ",sdr[0]
    score2 = pesq(y_s, y_mix, sr)
    score2_stoi = stoi(y_s, y_mix, sr)
    #sdr_mix, sir_mix, sar_mix, popt_mix = bss_eval_sources(y_s, y_mix)
    print("noisy: ",score2," stoi: ",score2_stoi)#," n_sdr: ",sdr_mix[0]
    score = max(score1,score2)
    #sdr_mix, sir_mix, sar_mix, popt_mix = bss_eval_sources(y_s, y_mix)
    #print(score)
    sum_denoised += score1
    sum_stoi1 += score1_stoi
    sum_noisy += score2
    sum_stoi2 += score2_stoi
    #sum_sdr += sdr[0]
    #sum_n_sdr += sdr_mix[0]
    #sum += max(score1,score2)
    k=k+1
    #print(sum/k)
#print("k:",k)
#print("sum_denoised:",sum_denoised)
print("denoised PESQ: ",sum_denoised/k," STOI: ",sum_stoi1/k)#," sdr: ",sum_sdr/k
print("noisy PESQ: ",sum_noisy/k," STOI: ",sum_stoi2/k)#," sdr_n: ",sum_n_sdr/k
    #wav_path_c = in_dir_c + wav_file_c  

    
    #y_c, sr = librosa.load(wav_path_c,mono=None, sr=None) 
    
