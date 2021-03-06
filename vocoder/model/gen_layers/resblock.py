import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import List


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: List[int], relu_slope: float = 1e-1):
        super().__init__()
        padding = list((kernel_size - 1) * d // 2 for d in dilation)
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, padding=padding[i], dilation=dilation[i]))
            for i in range(len(dilation))
        ])

    def forward(self, x):
        for conv in self.convs:
            res = x
            x = F.leaky_relu(x, 0.1)
            x = conv(x)
            x = x + res
        return x
