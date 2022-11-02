import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram, MelSpectrogram
import numpy as np
from tqdm import tqdm
from einops import rearrange
from collections import OrderedDict

class IPDExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_mag  = torch.log10(torch.abs(x)+1e-6)
        x_pha  = torch.angle(x)
        cospha = torch.cos(x_pha)
        sinpha = torch.sin(x_pha)
        x      = rearrange(x, 'b m f t -> b m 1 f t')
        x_conj = rearrange(torch.conj(x), 'b m 1 f t -> b 1 m f t')
        ipdmat = rearrange((x * x_conj).angle(), 'b m1 m2 f t -> b (m1 m2) f t')
        cosipd = torch.cos(ipdmat)
        sinipd = torch.sin(ipdmat)
        ipd    = torch.cat([x_mag, cospha, sinpha, cosipd, sinipd], -3)
        return ipd

    def sample_points(self, num_sampling_points):
        points = np.random.randn(num_sampling_points, 3)
        norm = np.sqrt(np.sum(points**2, -1, keepdim = True))
        points[:, 2] = np.abs(points[:, 2])
        points = points/norm
        return points

class ConvBlock(nn.Module):
    def __init__(self, c_i, c_o, k, s, depthwise=True):
        super().__init__()
        if depthwise:
            self.net = [nn.Conv2d(c_i, c_o, 1, 1, 0), nn.BatchNorm2d(c_o), nn.ReLU6(), 
                        nn.Conv2d(c_o, c_o, (k, 1), (s, 1), (k // 2, 0), groups = c_o), nn.BatchNorm2d(c_o), nn.ReLU6()]
        else:
            self.net = [nn.Conv2d(c_i, c_o, (k, 1), (s, 1), (k // 2, 0)), nn.BatchNorm2d(c_o), nn.ReLU6()]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)

class FConvNet(nn.Module):
    def __init__(self, in_features, out_features, tgru_size=896, lightweight=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.tconv = nn.Sequential(nn.Conv1d(in_features, 128, 5, 2, 2), nn.ReLU(),
                                   nn.Conv1d(128, 128, 5, 2, 2), nn.ReLU(),
                                   nn.Conv1d(128, 128, 5, 2, 2), nn.ReLU(),
                                   nn.Conv1d(128, 128, 5, 2, 2), nn.ReLU())

        if lightweight:
            self.fconv = nn.Sequential(*[ConvBlock(*args) for args in [(128, 128, 5, 2, True),
                                                                       (128, 128, 5, 2, True),
                                                                       (128, 128, 5, 2, True),
                                                                       (128, 128, 5, 2, True),
                                                                       (128, 128, 5, 2, True),
                                                                       (128, 128, 3, 2, True)]])
        else:
            self.fconv = nn.Sequential(*[ConvBlock(*args) for args in [(128, 128, 5, 2, False),
                                                                       (128, 128, 3, 1, False),
                                                                       (128, 128, 5, 2, False),
                                                                       (128, 128, 3, 1, False),
                                                                       (128, 128, 5, 2, False),
                                                                       (128, 128, 3, 1, False),
                                                                       (128, 128, 5, 2, False),
                                                                       (128, 128, 3, 1, False),
                                                                       (128, 128, 5, 2, False),
                                                                       (128, 128, 3, 2, False)]])

        self.tgru = nn.GRU(tgru_size, 512, num_layers=2, batch_first=True)
        self.linear = nn.Linear(512, out_features)

        self.h = None

    def forward(self, x):
        b = x.shape[0]
        x = rearrange(x, 'b c f t -> (b f) c t')
        x = self.tconv(x)
        x = rearrange(x, '(b f) c t -> b c f t', b = b)
        x = self.fconv(x)
        x = rearrange(x, 'b c f t -> b t (f c)', b = b)
        x, self.h = self.tgru(x, self.h.detach() if type(self.h) == torch.Tensor else self.h)
        x = self.linear(x)
        return x

class FeatureFrontEnd(nn.Module):
    def __init__(self, sr=16000, mel=False, ipd=False, n_channels=1):
        super().__init__()
        if sr == 16000:
            if mel:
                self.spec = MelSpectrogram(n_fft=400, hop_length=100, n_mels=80, center=False)
                self.in_features, self.tgru_size = 1, 256
            else:
                if ipd:
                    self.spec = Spectrogram(n_fft=400, hop_length=100, power=None, center=False)
                    self.in_features, self.tgru_size = n_channels*3+2*n_channels**2, 512
                else:
                    self.spec = Spectrogram(n_fft=400, hop_length=100, power=2, center=False)
                    self.in_features, self.tgru_size = 1, 512
        elif sr == 48000:
            if mel:
                self.spec = MelSpectrogram(n_fft=1200, hop_length=300, n_mels=180, center=False)
                self.in_features, self.tgru_size = 1, 384
            else:
                if ipd:
                    self.spec = Spectrogram(n_fft=1200, hop_length=300, power=None, center=False)
                    self.in_features, self.tgru_size = n_channels*3+2*n_channels**2, 1280
                else:
                    self.spec = Spectrogram(n_fft=1200, hop_length=300, power=2, center=False)
                    self.in_features, self.tgru_size = 1, 1280

        self.mel, self.ipd = mel, ipd
        if ipd: self.ipd_extractor = IPDExtractor()

    def forward(self, x):
        if (not self.mel) and self.ipd:
            return self.ipd_extractor(self.spec(x))
        else:
            return torch.log10(self.spec(x)+1e-6)

class Estimator(nn.Module):
    def __init__(self, lightweight=False, **features_kwargs):
        super().__init__()
        self.features = FeatureFrontEnd(**features_kwargs)
        self.cla_net = FConvNet(self.features.in_features, 3, self.features.tgru_size, lightweight=lightweight)
 
    def forward(self, x, skip_frontend=False):
        if not skip_frontend: x = self.features(x)
        return self.cla_net(x)

    def reset_state(self):
        self.cla_net.h = None

if __name__ == '__main__':
    # model 1
    net = Estimator(sr=16000, mel=True)
    print(net(torch.rand(1, 1, 16000)).shape)

    # model 2
    net = Estimator(sr=48000, mel=True)
    print(net(torch.rand(1, 1, 48000)).shape)

    # model 3
    net = Estimator(sr=16000, mel=False, ipd=False)
    print(net(torch.rand(1, 1, 16000)).shape)

    # model 4
    net = Estimator(sr=48000, mel=False, ipd=False)
    print(net(torch.rand(1, 1, 48000)).shape)

    # model 5
    net = Estimator(sr=16000, mel=False, ipd=True, n_channels=7)
    print(net(torch.rand(1, 7, 16000)).shape)
    
    # model 6
    net = Estimator(sr=48000, mel=False, ipd=True, n_channels=7)
    print(net(torch.rand(1, 7, 48000)).shape)
