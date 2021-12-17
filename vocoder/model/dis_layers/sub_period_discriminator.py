import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class SubPDiscriminator(nn.Module):
    def __init__(self, p, kernel_size: int = 5, stride: int = 3, relu_slope: float = 1e-1):
        super(SubPDiscriminator, self).__init__()
        self.p = p
        padding = (kernel_size - 1) // 2
        channels = 32

        layers = [weight_norm(nn.Conv2d(
            1, channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        ))]

        channels = [32, 128, 512, 1024]
        for i in range(3):
            layers += [weight_norm(nn.Conv2d(
                channels[i], channels[i + 1],
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0)
            ))]

        channels = channels[-1]
        layers += [
            weight_norm(nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(channels, 1, kernel_size=(3, 1), padding=(1, 0)))
        ]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        _, channels, time = x.size()
        if time % self.p != 0:
            n_pad = self.p - (time % self.p)
            x = F.pad(x, (0, n_pad), "reflect")
            time = x.size(-1)
        x = x.view(-1, channels, time // self.p, self.p)

        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 != len(self.layers):
                x = F.leaky_relu(x, 0.1)
            features.append(x)

        return torch.flatten(x, 1, -1), features
