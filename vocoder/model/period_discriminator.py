import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .dis_layers import SubPDiscriminator


class MPD(nn.Module):
    def __init__(self, periods: List[int], kernel_size: int = 5, stride: int = 3, relu_slope: float = 1e-1):
        super(MPD, self).__init__()
        self.periods = periods
        self.subs = nn.ModuleList([
            SubPDiscriminator(period, kernel_size, stride, relu_slope) for period in periods
        ])

    def forward(self, wav, wav_pred):
        wavs, wavs_pred = [], []
        features, features_pred = [], []

        for sub in self.subs:
            mpd_real, feature = sub(wav)
            wavs.append(mpd_real)
            features.append(feature)

            mpd_pred, feature_pred = sub(wav_pred)
            wavs_pred.append(mpd_pred)
            features_pred.append(feature_pred)
        return wavs, wavs_pred, features, features_pred
