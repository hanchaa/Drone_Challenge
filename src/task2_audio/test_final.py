import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio 
import os
from torch.utils.data import Dataset 
from torch.utils.data.dataloader import DataLoader
from einops import rearrange, repeat

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from models import Estimator
from datetime import datetime
from validation_utils_16k_new import *
import os
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type = str, default = '/home/drone/Desktop/gc_2022_1/ckpts/16k_magspec.0080.pt')
parser.add_argument('--filename', type = str, default = None)
parser.add_argument('--threshold', type = float, default = None)
parser.add_argument('--smooth', type = int, default = None)
parser.add_argument('--merge_frame', type = int, default = None)
parser.add_argument('--min_frame', type = int, default = None)
parser.add_argument('--mel', dest='mel', action='store_true')
parser.set_defaults(mel=False)
parser.add_argument('--ipd', dest='ipd', action='store_true')
parser.set_defaults(ipd=False)
parser.add_argument('--lightweight', dest='lightweight', action='store_true')
parser.set_defaults(lightweight=False)
parser.add_argument('--n_channels', type=int, default=1)
parser.add_argument('--sr', type=int, default=16000)
parser.add_argument('--debug', type=bool, default=False)
args = parser.parse_args()

ckpt = torch.load(args.checkpoint, map_location='cpu')
estimator = Estimator(lightweight=args.lightweight, sr=args.sr, mel=args.mel, ipd=args.ipd, n_channels=args.n_channels)
estimator.load_state_dict(ckpt['net'])
estimator.eval()
# res_log = test(estimator, args.threshold, args.smooth, args.min_frame, args.merge_frame)
flag = True
audio_path = '/home/drone/Desktop/wavwavwavwav.wav' # this is made by running Desktop/7.sh
index = 0
print('start inference')

current_index = 0
duration = args.sr * 1 # put duration here
hop_size = duration // 2

while flag:
    wav_file = audio_path
    #try:
    if os.path.isfile(wav_file):
        try:
            y,sr = sf.read(audio_path,start=current_index,stop=current_index+duration)
            assert y.shape[0] == duration
            y = np.transpose(y)
            y = np.mean(y,axis=0)
            audio = torch.Tensor(y)
            audio = audio.unsqueeze(0).unsqueeze(0)
            start = time.time()
            print(f'results: {estimator(audio).permute(0,2,1).argmax(dim=1)}')
            end = time.time()
            print(f'time elapsed: {end-start}')
            index += 1
            current_index += hop_size
        except AssertionError:
            if args.debug: print(f"audio not yet ready. current index is {current_index}")
            time.sleep(0.1)
        except SystemError as e:
            print(e)
            time.sleep(0.5)
    else:
        time.sleep(0.1)
    #except:
    #    continue 

