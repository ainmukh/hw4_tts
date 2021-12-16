import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: List[int], relu_slope: float = 1e-1):
        super().__init__()

        padding = list((kernel_size - 1) * d // 2 for d in dilation)
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, (kernel_size, 1), padding=padding[i], dilation=dilation[i])
            for i in range(len(dilation))
        ])
        self.relu_slope = relu_slope

    def forward(self, x):
        for conv in self.convs:
            res = x
            x = F.leaky_relu(x, self.relu_slope)
            x = conv(x)
            x = x + res
        return x
