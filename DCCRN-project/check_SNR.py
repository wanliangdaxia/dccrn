import numpy as np
import librosa
 
 
#计算信噪比
def SNR_singlech(clean_file, original_file):
    clean, clean_fs = librosa.load(clean_file, sr=None, mono=True)#导入干净语音
    est_noise, ori_fs = librosa.load(original_file, sr=None, mono=True)#导入原始语音
    #length = min(len(clean), len(est_noise))
   # est_noise = ori[:length] - clean[:length]#计算噪声语音
    
    #计算信噪比
    SNR = 10*np.log10((np.sum(clean**2))/(np.sum(est_noise**2)))
    print(SNR)
 
SNR_singlech('/root/howl/howl_-5dB/clean/1088.wav', '/root/howl/howl_-5dB/noise/1088.wav')