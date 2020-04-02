import torch
from torch import nn
from architectures.NVDLMED.model.GroupNorm import *


class ResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResNetBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.skip_connection = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1, 1, 1), stride=1)

        self.direct_connection = nn.Sequential(
            GroupNorm(in_channel, num_groups=8),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3, 3), stride=1),
            GroupNorm(in_channel, num_groups=8),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3, 3), stride=1)
        )

    def forward(self, x):
        skip_connection_out = self.direct_connection(x)
        direct_out = self.layers(x)

        out = skip_connection_out + direct_out

        return out

