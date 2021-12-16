import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .gen_layers import UpSampleBlock


class Generator(nn.Module):
    def __init__(self,
                 mel: int, pre_channels: int, kernel_size: List[int],
                 kernel_res: List[int], dilation: List[List[List[int]]], relu_slope: float = 1e-1):
        super().__init__()
        self.pre_conv = nn.Conv1d(mel, pre_channels, (7, 1), padding=3)

        self.upsample = nn.Sequential(*list(
            UpSampleBlock(pre_channels // 2**i, kernel_size[i], kernel_res, dilation[i], relu_slope)
            for i in range(len(kernel_size))
        ))

        self.post_conv = nn.Conv1d(pre_channels // 2**len(kernel_size), 1, (7, 1), padding=3)
        self.relu_slope = relu_slope

    def forward(self, x):
        x = self.post_conv(x)

        x = self.upsample(x)

        x = F.leaky_relu(x, self.relu_slope)
        x = self.post_conv(x)
        x = F.tanh(x)
        return x
