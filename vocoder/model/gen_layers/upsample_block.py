import torch.nn as nn
import torch.nn.functional as F
from typing import List
from ..gen_layers import MultiReceptiveField


class UpSampleBlock(nn.Module):
    def __init__(self,
                 channels: int, kernel_size: int,
                 kernel_res: List[int], dilation: List[List[List[int]]], relu_slope: float = 1e-1):
        super(UpSampleBlock, self).__init__()
        stride = kernel_size // 2
        padding = (kernel_size - stride) // 2
        self.conv = nn.ConvTranspose1d(channels, channels // 2, kernel_size, stride, padding=padding)
        self.mrf = MultiReceptiveField(kernel_res, dilation, relu_slope, channels // 2)
        self.relu_slope = relu_slope

    def forward(self, x):
        x = F.leaky_relu(x, self.relu_slope)
        x = self.conv(x)
        # print('after up', x.size())
        x = self.mrf(x)
        # exit(print('after res', x.size()))
        return x
