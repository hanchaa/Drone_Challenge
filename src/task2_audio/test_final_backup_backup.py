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
args = parser.parse_args()

ckpt = torch.load(args.checkpoint, map_location='cpu')
estimator = Estimator(lightweight=args.lightweight, sr=args.sr, mel=args.mel, ipd=args.ipd, n_channels=args.n_channels)
estimator.load_state_dict(ckpt['net'])
estimator.eval()
# res_log = test(estimator, args.threshold, args.smooth, args.min_frame, args.merge_frame)
flag = True
audio_path = '/home/drone/Desktop/wavfile_dest_for_4'
index = 0
print('start inference')
while flag:
    wav_file = os.path.join(audio_path,f'{index}.wav')
    try:
        if os.path.isfile(wav_file):
            audio, sr = torchaudio.load(wav_file)
            print(audio.shape)
            audio = audio.unsqueeze(0)
            start = time.time()
            print(f'results: {estimator(audio).permute(0,2,1).argmax(dim=1)}')
            end = time.time()
            print(f'time elapsed: {end-start}')
            index += 1
        else:
            continue
    except:
        continue 

