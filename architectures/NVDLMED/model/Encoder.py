from torch import nn
from architectures.NVDLMED.model.ResNetBlock import *


class Encoder(nn.Module):
    def __init__(self, input_shape=(4, 160, 192, 128)):
        super(Encoder, self).__init__()

        self.input_shape = input_shape

        self.initial_block = nn.Sequential(nn.Conv3d(in_channels=self.input_shape[0], out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=1),
                                           nn.Dropout3d(p=0.2))

        self.first_encoder = ResNetBlock(in_channel=32, out_channel=32)

        self.second_encoder = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=2, padding=1),
                                            ResNetBlock(in_channel=32, out_channel=64),
                                            ResNetBlock(in_channel=64, out_channel=64))

        self.third_encoder = nn.Sequential(nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=2, padding=1),
                                           ResNetBlock(in_channel=64, out_channel=128),
                                           ResNetBlock(in_channel=128, out_channel=128))

        self.fourth_encoder = nn.Sequential(nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=2, padding=1),
                                            ResNetBlock(in_channel=128, out_channel=256),
                                            ResNetBlock(in_channel=256, out_channel=256),
                                            ResNetBlock(in_channel=256, out_channel=256),
                                            ResNetBlock(in_channel=256, out_channel=256))

    def forward(self, x):
        # Initial Block
        x = self.initial_block(x)

        # First Encoder ResNetBlock (output filters = 32)
        x1 = self.first_encoder(x)

        # Second Encoder ResNetBlock (output filters = 64)
        x2 = self.second_encoder(x1)

        # Third Encoder ResNetBlock (output filters = 128)
        x3 = self.third_encoder(x2)

        # Fourth Encoder ResNetBlock (output filters = 256)
        x4 = self.fourth_encoder(x3)

        return x1, x2, x3, x4
