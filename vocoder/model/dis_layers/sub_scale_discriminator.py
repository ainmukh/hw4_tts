import torch
import torch.nn as nn
import torch.nn.functional as F


class SubSDiscriminator(nn.Module):
    def __init__(self, pool: int = 0, kernel_size: int = 41, relu_slope: float = 1e-1, groups: int = 16):
        super(SubSDiscriminator, self).__init__()
        padding = (pool - 1) // 2
        self.pool = nn.AvgPool1d(pool, pool // 2, padding=padding) if pool else nn.Identity()
        padding = (kernel_size - 1) // 2
        channels = 128
        stride = [2, 4, 4]

        layers = [
            nn.Conv1d(1, channels, kernel_size=15, stride=1, padding=7),
            # nn.Conv1d(channels, channels, kernel_size, stride=2, groups=4, padding=padding),
        ]
        for i in range(4):
            layers += [nn.Conv1d(
                channels, channels * 2,
                kernel_size, stride[i],
                groups=groups, padding=padding
            )]
            channels = channels * 2
        layers += [
            nn.Conv1d(channels, channels, kernel_size, 1, groups=groups, padding=padding),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.Conv1d(channels, 1, kernel_size=3, padding=1)
        ]

        self.layers = nn.ModuleList(layers)
        self.relu_slope = relu_slope

    def forward(self, x):
        x = self.pool(x)

        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 != len(self.layers):
                x = F.leaky_relu(x, self.relu_slope)
            features.append(x)
        return torch.flatten(x, 1, -1), features
