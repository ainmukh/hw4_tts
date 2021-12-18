import torch
import torch.nn as nn
import torch.nn.functional as F
from .dis_layers import SubSDiscriminator


class MSD(nn.Module):
    def __init__(self, kernel_size: int = 41, relu_slope: float = 1e-1, groups: int = 16):
        super().__init__()
        self.subs = nn.ModuleList([
            SubSDiscriminator(2**i, kernel_size, relu_slope, groups) for i in range(3)
        ])

    # def forward(self, wav):
    #     outputs = []
    #     features = []
    #
    #     for sub in self.subs:
    #         output, feature = sub(wav)
    #         outputs.append(output)
    #         features.append(feature)
    #
    #     return outputs, features

    def forward(self, wav, wav_pred):
        wavs, wavs_pred = [], []
        features, features_pred = [], []

        for sub in self.subs:
            msd_real, feature = sub(wav)
            wavs.append(msd_real)
            features.append(feature)

            msd_pred, feature_pred = sub(wav_pred)
            wavs_pred.append(msd_pred)
            features_pred.append(feature_pred)
        return wavs, wavs_pred, features, features_pred
