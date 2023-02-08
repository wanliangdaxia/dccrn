# coding: utf-8
# Author：WangTianRui
# Date 2020/9/29 21:47
import torch.nn as nn
import wav_loader_c as loader
import net_config as net_config
import pickle
from torch.utils.data import DataLoader
#import module as model_cov_bn
from si_snr import *
import train_utils
import os
#from memory_profiler import profile
import  model as model
from solver import Solver
########################################################################
# Change the path to the path on your computer
dns_home = "/root/text/"  # dir of dns-datas  #r"F:\Traindata\DNS-Challenge\make_data"
save_file = "./logs"  # model save
########################################################################

batch_size = 160  # calculate batch_size
load_batch = 6 # load batch_size(not calculate)


lr = 0.0001  # learning_rate
with open('/root/rain/data/train_noisy.txt', 'r') as train_file_noisy:
    train_noisy_names = [line.strip() for line in train_file_noisy.readlines()]

with open('/root/rain/data/train_noise.txt', 'r') as train_file_clean:
    train_clean_names = [line.strip() for line in train_file_clean.readlines()]

with open('/root/rain/data/valid_noisy.txt', 'r') as valid_file_noisy:
    test_noisy_names = [line.strip() for line in valid_file_noisy.readlines()]

with open('/root/rain/data/valid_noise.txt', 'r') as valid_file_clean:
    test_clean_names = [line.strip() for line in valid_file_clean.readlines()]

train_dataset = loader.WavDataset(train_noisy_names, train_clean_names, frame_dur=37.5) #37.5->40
test_dataset = loader.WavDataset(test_noisy_names, test_clean_names, frame_dur=37.5)
# dataloader
print("make_dataloader: start")
tr_loader = loader.AudioDataLoader(train_dataset, batch_size=load_batch, shuffle=True, num_workers = 0)
cv_loader = loader.AudioDataLoader(test_dataset, batch_size=load_batch, shuffle=True, num_workers = 0)
  
data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"#指定某个GPU
model = model.DCCRN()
model = nn.DataParallel(model, device_ids=[0])

#指定GPU进行计算
import os 


model.cuda()#调用model.cuda()，可以将模型加载到GPU上去


k = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('# of parameters:', k)
optimizier = torch.optim.Adam(model.parameters(),
                                    lr=lr)#调整模型的参数和学习率
                                
solver = Solver(data, model, optimizier)#将数据、模型、优化器给到Solver
solver.train()


