# coding: utf-8
# AuthorangTianRui
# Date 2020/9/29 11:03

from torch.utils.data import Dataset, DataLoader
import librosa #as lib
import os
import numpy as np
import torch
from scipy import fftpack 
import torch.utils.data as data



class WavDataset(Dataset):
    def __init__(self, noisy_paths, clean_paths, frame_dur=37.5):
        self.noisy_paths = noisy_paths #带噪语音路径
        self.clean_paths = clean_paths #干净语音路径

        self.frame_dur = frame_dur        
        minibatch = []
        sample_rate = 16000
        segment = 16000     
        for noisy_file, clean_file in zip(self.noisy_paths,self.clean_paths):     
            part_mix = []
            part_s1 = []
            part_mix.append(noisy_file)
            part_s1.append(clean_file)
            minibatch.append([part_mix,
                          part_s1,
                          sample_rate, 
                          segment])
        self.minibatch = minibatch
    def __getitem__(self, item):
        return self.minibatch[item]
    def __len__(self):
        return len(self.minibatch)

class AudioDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
        
def _collate_fn(batch):
    mixtures, sources = load_mixtures_and_sources(batch[0])
    clean_file = []
    pad_value = 0
    mixtures_pad_end = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures],pad_value)
    sources_pad_end = pad_list([torch.from_numpy(s).float()
                            for s in sources], pad_value)
    clean_file.append(batch[0][0])


    for i in range(len(batch)-1):

        mixtures, sources = load_mixtures_and_sources(batch[i+1])

    #ilens = np.array([mix.shape[0] for mix in mixtures])
        #print(sources[1].shape)
        assert sources[0].shape[1]==sources[1].shape[1]
    # perform padding and convert to tensor
        pad_value = 0
        #mixtures_pad = torch.from_numpy(mixtures).float()
        #sources_pad = torch.from_numpy(sources).float()
        
        mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures],pad_value)
        
        sources_pad = pad_list([torch.from_numpy(s).float()
                            for s in sources], pad_value)
        
        mixtures_pad_end = torch.cat((mixtures_pad_end,mixtures_pad), 0) 
        sources_pad_end =  torch.cat((sources_pad_end,sources_pad), 0)
        clean_file.append(batch[i+1][0])
    # N x T x C -> N x C x T
    #sources_pad = sources_pad.permute((0, 2, 1)).contiguous()
    # get batch of lengths of input sequences
    #print(mixtures_pad_end.size())
    return mixtures_pad_end, sources_pad_end, clean_file


def load_mixtures_and_sources(batch):
    mixtures, sources = [], []
    mix_infos, s1_infos, sample_rate, segment_len = batch
    for mix_info, s1_info in zip(mix_infos, s1_infos):
        mix_path = mix_info
        s1_path = s1_info
        assert mix_info[1] == s1_info[1]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)

        s1, _ = librosa.load(s1_path, sr=sample_rate)
        ##mix_pad = Hilbert(mix)
        ##s1_pad = Hilbert(s1) 
        mix_pad = mix.reshape(1,-1)
        s1_pad = s1.reshape(1,-1)    
        utt_len = mix.shape[-1]
        if segment_len >= 0:
            # segment
            for i in range(0, utt_len - segment_len + 1, segment_len):
                mixtures.append(mix_pad[:,i:i+segment_len])
                sources.append(s1_pad[:,i:i+segment_len])
            if utt_len % segment_len != 0:
                mixtures.append(mix_pad[:,-segment_len:])
                sources.append(s1_pad[:,-segment_len:])
        else:  # full utterance
            mixtures.append(mix)
            sources.append(s)
    #print(len(mixtures))
    return mixtures, sources



def pad_list(xs, pad_value):
    n_batch = len(xs)
    #print(n_batch)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

if __name__ == '__main__':
    a = np.random.rand(1)
    b = fftpack.hilbert(a)
    print(a)