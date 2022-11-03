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


def update_result(initial_result):
    if initial_result["male"] == 1:
        self.result["male"] = initial_result["male"]
    if initial_result["female"] == 1:
        self.result["female"] = initial_result["female"]
    if initial_result["baby"] == 1:
        self.result["baby"] = initial_result["baby"]


def write_csv(state, result):
    file_path = os.path.join("./task2_audio/results", str(state) + ".csv")
    file_mode = "a" if os.path.isfile(file_path) else "w"

    with open(file_path, file_mode, newline="") as f:
        wr = csv.writer(f)
        wr.writerow([result["male"], result["female"], result["baby"]])


def get_setting(ckpt_path):
    if os.path.basename(ckpt_path) == '16k_magspec.0080.pt':
        return float(0.6), 3, 28, 22
    elif os.path.basename(ckpt_path) == '16k_magspec.0030.pt':
        return 0.6, 3, 20, 12
    elif os.path.basename(ckpt_path) == '16k_magspec.light.0050.pt':
        return 0.64, 7, 12, 18
    elif os.path.basename(ckpt_path) == '16k_magspec.light.0030.pt':
        return 0.63, 5, 20, 16
    else:
        raise Exception('InvalidPathError')


class Task2Audio():
    def __init__(self, args, pub):
        self.pub = pub

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        self.estimator = Estimator(lightweight=args.lightweight, sr=args.sr, mel=args.mel, ipd=args.ipd, n_channels=args.n_channels)
        self.estimator.load_state_dict(ckpt["net"])
        self.estimator = self.estimator.to("cuda" if torch.cuda.is_available() else "cpu")
        self.estimator.eval()

        self.threshold, self.smooth, self.merge_frame, self.min_frame = get_setting(args.checkpoint)
        self.audio_path = os.path.join("./task2_audio", args.filename)

        self.current_index = 0
        self.duration = args.sr * 1
        self.hop_size = duration // 2

        self.prev_state = -1
        self.result = {"male": 0, "female": 0, "baby": 0}

    def move(self, x, y):
        plt.scatter(x, y, color="red")

        goal = PoseStamped()

        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 1.4
        goal.pose.orientation.w = 1.0

        self.pub.publish(goal)

    def __call__(self, state):
        if os.path.isfile(self.audio_path):
            try:
                y, xr = sf.read(self.audio_path, start=self.current_index, stop=self.current_index + self.duration)
                assert y.shape[0] == self.duration

                y = np.transpose(y)
                y = np.mean(y, axis=0)
                audio = torch.Tensor(y)
                audio = audio.unsqueeze(0).unsqueeze(0)
                audio = audio.to(device)

                initial_result = test(self.estimator, audio, self.threshold, self.smooth, self.min_frame, self.merge_frame)
                update_result(initial_result)

                if self.prev_state != state:
                    write_csv(state, self.result)
                    self.prev_state = state

                    if state == 1:
                        move(-6.11, -17.6)
                    elif state == 3:
                        move(-6.09, -10.94)
                    elif state == 5:
                        move(-6.17, -4.46)

                self.current_index += self.hop_size

            except AssertionError:
                if args.debug:
                    print(f"audio not yet ready. current index is {self.current_index}")
                    time.sleep(0.1)
            except SystemError as e:
                print(e)
                time.sleep(0.5)

        else:
            time.sleep(0.1)