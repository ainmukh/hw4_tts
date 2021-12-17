import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class SubSDiscriminator(nn.Module):
    def __init__(self, pool: int = 0, kernel_size: int = 41, relu_slope: float = 1e-1, groups: int = 16):
        super(SubSDiscriminator, self).__init__()
        self.pool = nn.AvgPool1d(4, 2, padding=2) if pool else None
        padding = (kernel_size - 1) // 2
        channels = 128
        stride = [2, 4, 4]

        layers = [
            weight_norm(nn.Conv1d(1, channels, kernel_size=15, stride=1, padding=7))
            # nn.Conv1d(channels, channels, kernel_size, stride=2, groups=4, padding=padding),
        ]
        for i in range(3):
            layers += [weight_norm(nn.Conv1d(
                channels, channels * 2,
                kernel_size, stride[i],
                groups=groups, padding=padding
            ))]
            channels = channels * 2
        layers += [
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=groups, padding=padding)),
            weight_norm(nn.Conv1d(channels, channels, kernel_size=5, padding=2))
        ]
        self.post = weight_norm(nn.Conv1d(channels, 1, kernel_size=3, padding=1))

        self.layers = nn.ModuleList(layers)
        # exit(print(self.layers))

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)

        features = []
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        x = self.post(x)
        features.append(x)

        return torch.flatten(x, 1, -1), features
