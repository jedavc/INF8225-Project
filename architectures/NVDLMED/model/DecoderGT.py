from torch import nn
from architectures.NVDLMED.model.ResNetBlock import *


class DecoderGT(nn.Module):
    def __init__(self, input_channels=256, output_channels=3):
        super(DecoderGT, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.first_upsample3d = nn.Sequential(nn.Conv3d(in_channels=input_channels, out_channels=128, kernel_size=(1, 1, 1), stride=1),
                                              nn.Upsample(scale_factor=2, mode='trilinear'))

        self.first_resnetblock = ResNetBlock(in_channel=128, out_channel=128)

        self.second_upsample3d = nn.Sequential(nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(1, 1, 1), stride=1),
                                               nn.Upsample(scale_factor=2, mode='trilinear'))

        self.second_resnetblock = ResNetBlock(in_channel=64, out_channel=64)

        self.third_upsample3d = nn.Sequential(nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(1, 1, 1), stride=1),
                                              nn.Upsample(scale_factor=2, mode='trilinear'))

        self.third_resnetblock = ResNetBlock(in_channel=32, out_channel=32)

        self.output_gt = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=1),
                                       nn.Conv3d(in_channels=32, out_channels=self.output_channels, kernel_size=(1, 1, 1), stride=1))

    def forward(self, x1, x2, x3, x4):
        # First Decoder ResNetBlock (output filters = 128)
        x = self.first_upsample3d(x4)
        x = torch.add(x, x3)
        x = self.first_resnetblock(x)

        # Second Decoder ResNetBlock (output filters = 64)
        x = self.second_upsample3d(x)
        x = torch.add(x, x2)
        x = self.second_resnetblock(x)

        # Third Decoder ResNetBlock (output filters = 32)
        x = self.third_upsample3d(x)
        x = torch.add(x, x1)
        x = self.third_resnetblock(x)

        # Output Block
        out_GT = self.output_gt(x)

        return out_GT
