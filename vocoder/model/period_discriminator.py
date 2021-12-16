import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .dis_layers import SubPDiscriminator


class MPD(nn.Module):
    def __init__(self, periods: List[int]):
        super(MPD, self).__init__()
        self.periods = periods
        self.subs = nn.ModuleList([
            SubPDiscriminator(period) for period in periods
        ])

    def forward(self, wav, wav_pred):
        wavs, wavs_pred = [], []
        features, features_pred = [], []

        for sub in self.subs:
            wav, feature = sub(wav)
            wavs.append(wav)
            features.append(feature)

            wav_pred, feature_pred = sub(wav_pred)
            wavs_pred.append(wav_pred)
            features_pred.append(feature_pred)
        return wavs, wavs_pred, features, features_pred
