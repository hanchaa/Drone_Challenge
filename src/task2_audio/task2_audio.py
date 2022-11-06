import os
import csv
import time
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from einops import rearrange, repeat
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
import soundfile as sf

from .models import Estimator
from .utils import *


def get_setting(ckpt_path):
    if os.path.basename(ckpt_path) == '16k_magspec.0080.pt':
        return float(0.6), 3, 28, 22
    elif os.path.basename(ckpt_path) == '16k_magspec.0030.pt':
        return 0.6, 3, 20, 12
    elif os.path.basename(ckpt_path) == '16k_magspec.light.0050.pt':
        return 0.64, 7, 12, 18 
    elif os.path.basename(ckpt_path) == '16k_magspec.light.0030.pt':
        return 0.63, 5, 20, 16
    elif os.path.basename(ckpt_path) == '2022_11_03_17_47_47.0100.pt':
        return 0.71, 3, 28, 10
    elif os.path.basename(ckpt_path) == '2022_11_03_17_49_30.0110.pt':
        return 0.6, 5, 20, 12
    else:
        raise Exception('InvalidPathError')


class Task2Audio():
    def __init__(self, args):
        self.args = args

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        self.estimator = Estimator(lightweight=args.lightweight, sr=args.sr, mel=True, ipd=args.ipd, n_channels=args.n_channels)
        self.estimator.load_state_dict(ckpt["net"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.estimator = self.estimator.to(self.device)
        self.estimator.eval()

        self.threshold, self.smooth, self.merge_frame, self.min_frame = get_setting(args.checkpoint)
        self.audio_path = os.path.join("./task2_audio", args.filename)

        self.current_index = 0
        self.duration = args.sr * 10
        self.hop_size = args.sr // 2
        self.start_index = 0

        self.prev_state = -1
        self.result = {"male": 0, "female": 0, "baby": 0}

    def __call__(self, state):
        if os.path.isfile(self.audio_path):
            try:

                if self.prev_state != state:
                    
                    print(f"{self.prev_state}: {self.result}")
                    self.prev_state = state
                    self.result = {"male": 0, "female": 0, "baby": 0}
                    self.current_index = torchaudio.info(self.audio_path)[0].length
    
                y, xr = sf.read(self.audio_path, start=self.current_index, stop=self.current_index + self.duration)
                assert y.shape[0] > 16000 * 2
                
                if y.shape[0] > self.duration:
                    self.current_index += self.hop_size
                    y = y[-self.duration:,:]                

                y = np.transpose(y)
                y = np.mean(y, axis=0)
                audio = torch.Tensor(y)
                audio = audio.unsqueeze(0).unsqueeze(0)
                audio = audio.to(self.device)

                initial_result = test(self.estimator, audio, self.threshold, self.smooth, self.min_frame, self.merge_frame)
                self.update_result(initial_result)

                

            except AssertionError:
                if self.args.debug:
                    print(f"audio not yet ready. current index is {self.current_index}")
                    time.sleep(0.1)
            except SystemError as e:
                print(e)
                time.sleep(0.5)

            return self.result

        else:
            time.sleep(0.1)
            return self.result

    def update_result(self, initial_result):
        if initial_result["male"] == 1:
            self.result["male"] = initial_result["male"]
        if initial_result["female"] == 1:
            self.result["female"] = initial_result["female"]
        if initial_result["baby"] == 1:
            self.result["baby"] = initial_result["baby"]