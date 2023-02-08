#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script tests the execution time of the DTLN model on a CPU.
Please use TF 2.2 for comparability.
Just run "python measure_execution_time.py"
Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 13.05.2020
This code is licensed under the terms of the MIT-license.
"""

import numpy as np

import argparse
import os
import pickle
import librosa
import torch
from torch.utils.data import DataLoader
import wav_loader as loader
import train_utils
import numpy
import re
import model as model
import time
# only use the cpu
os.environ["CUDA_VISIBLE_DEVICES"]=''

if __name__ == '__main__':
    # loading model in saved model format
    model_dict = model.DCCRN().load_model('/root/Kind_Of_Net/DCCRN/DCCRN-Wu/logs/final.pth.tar')
    # mapping signature names to functions
    model_dict.eval()
        
    exec_time = []
    # create random input for testing
    for i in range(1010):

        x = np.random.randn(16000 * 6) * 0.001
        x = np.clip(x, -1, 1)
        # input = torch.randn([1,16000*4])
        x = torch.from_numpy(x[None, None, :].astype('float32'))
        x = x[:,:,:160]
        print(x.size())
        # run timer
        start_time = time.time()
        # infer one block
        y = model_dict(x)
        exec_time.append((time.time() - start_time))
    # ignore the first ten iterations
    print('Execution time per block: ' + 
          str( np.round(np.mean(np.stack(exec_time[10:]))*1000, 2)) + ' ms')

# Ubuntu 18.04          I5 6600k        @ 3.5 GHz:  0.65 ms (4 cores)
# Macbook Air mid 2012 	I7 3667U        @ 2.0 GHz:  1.4 ms  (2 cores)
# Raspberry Pi 3 B+     ARM Cortex A53  @ 1.4 GHz: 15.54    (4 cores)