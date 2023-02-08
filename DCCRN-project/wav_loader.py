# coding: utf-8
# Author：WangTianRui
# Date 2020/9/29 11:03

from torch.utils.data import Dataset, DataLoader
import librosa as lib
import os
import numpy as np
import torch


def load_wav(path, frame_dur, sr=8000): #16k->8k
    
    signal, _ = lib.load(path, sr=sr)
    #win = int(frame_dur / 1000 * sr) #win=600
    #print(signal.shape)
    date = signal.reshape(1,-1)
    tmp = []
    if(date.shape[1]<16000):
        date = np.pad(date,(0,16000-date.shape[1]),'constant',constant_values=(0,0))
    elif(date.shape[1]>16000):
        for i in range(0, date.shape[1] - 16000 + 1, 16000):
                #print(utt_len - segment_len + 1)
                tmp.append(date[:,i:i+16000])
        if date.shape[1] % 16000 != 0:
            tmp.append(date[:,-16000:])
        date = np.array(tmp)
        date = date.squeeze(1)        
    #print(date.shape)
    return torch.from_numpy(date)#torch.tensor(np.split(signal, int(len(signal) / win), axis=0))


class WavDataset(Dataset):
    def __init__(self, noisy_paths, clean_paths, loader=load_wav, frame_dur=37.5):
        self.noisy_paths = noisy_paths
        self.clean_paths = clean_paths
        self.loader = loader
        self.frame_dur = frame_dur

    def __getitem__(self, item):
        noisy_file = self.noisy_paths[item]
        clean_file = self.clean_paths[item]
        return self.loader(noisy_file, self.frame_dur), self.loader(clean_file, self.frame_dur), noisy_file

    def __len__(self):
        return len(self.noisy_paths)


def load_hop_wav(path, frame_dur, hop_dur, sr=16000):
    signal, _ = lib.load(path, sr=sr)
    win = int(frame_dur / 1000 * sr)
    hop = int(hop_dur / 1000 * sr)
    rest = (len(signal) - win) % hop
    signal = np.pad(signal, (0, hop - rest), "constant")
    n_frames = int((len(signal) - win) // hop)
    strides = signal.itemsize * np.array([hop, 1])
    return torch.tensor(np.lib.stride_tricks.as_strided(signal, shape=(n_frames, win), strides=strides))


class WavHopDataset(Dataset):
    def __init__(self, noisy_paths, clean_paths, frame_dur, hop_dur, loader=load_hop_wav):
        self.noisy_paths = noisy_paths
        self.clean_paths = clean_paths
        self.loader = loader
        self.frame_dur = frame_dur
        self.hop_dur = hop_dur

    def __getitem__(self, item):
        noisy_file = self.noisy_paths[item]
        clean_file = self.clean_paths[item]
        return self.loader(noisy_file, self.frame_dur, self.hop_dur), \
               self.loader(clean_file, self.frame_dur, self.hop_dur)

    def __len__(self):
        return len(self.noisy_paths)
