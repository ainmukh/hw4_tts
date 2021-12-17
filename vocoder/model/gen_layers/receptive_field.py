import torch.nn as nn
from typing import List
from ..gen_layers import ResBlock


class MultiReceptiveField(nn.Module):
    def __init__(self, kernel_size: List[int], dilation: List[List[List[int]]], relu_slope: float, channels: int):

        super(MultiReceptiveField, self).__init__()
        self.blocks = nn.ModuleList([
            ResBlock(channels, kernel_size[i], dilation[i], relu_slope) for i in range(len(kernel_size))
        ])

    def forward(self, x):
        out = 0
        for block in self.blocks:
            out += block(x)
        return out / len(self.blocks)
