from torch import nn
from architectures.NVDLMED.model.ResNetBlock import *


class DecoderGT(nn.Module):
    def __init__(self, output_channels=4):
        super(DecoderGT, self).__init__()
        self.output_channels = output_channels

        self.upsample3d = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x1, x2, x3, x4):

        # First Decoder ResNetBlock (output filters = 128)
        x = nn.Conv3d(in_channels=x4.shape[1], out_channels=128, kernel_size=(1, 1, 1), stride=1)(x4)
        x = self.upsample3d(x)
        x = x + x3
        x = ResNetBlock(in_channel=128, out_channel=128)(x)

        # Second Decoder ResNetBlock (output filters = 64)
        x = nn.Conv3d(in_channels=x.shape[1], out_channels=64, kernel_size=(1, 1, 1), stride=1)(x)
        x = self.upsample3d(x)
        x = x + x2
        x = ResNetBlock(in_channel=64, out_channel=64)(x)

        # Third Decoder ResNetBlock (output filters = 32)
        x = nn.Conv3d(in_channels=x.shape[1], out_channels=32, kernel_size=(1, 1, 1), stride=1)(x)
        x = self.upsample3d(x)
        x = x + x1
        x = ResNetBlock(in_channel=32, out_channel=32)(x)

        # Blue Decoder (output filters = 32)
        x = nn.Conv3d(in_channels=x.shape[1], out_channels=32, kernel_size=(3, 3, 3), stride=1)(x)

        # Output Block
        out_GT = nn.Conv3d(in_channels=x.shape[1], out_channels=self.output_channels, kernel_size=(1, 1, 1), stride=1)(x)

        return out_GT
