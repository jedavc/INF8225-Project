from torch import nn
from architectures.NVDLMED.model.ResNetBlock import *


class Encoder(nn.Module):
    def __init__(self, input_shape=(4, 160, 192, 128)):
        super(Encoder, self).__init__()

        self.input_shape = input_shape

    def forward(self, x):

        # Initial Block
        x = nn.Conv3d(in_channels=self.input_shape[0], out_channels=32, kernel_size=(3, 3, 3), stride=1)(x)
        x = nn.Dropout3d(p=0.2)(x)

        # First ResNetBlock (output filters = 32)
        x1 = ResNetBlock(in_channel=32, out_channel=32)(x)
        x = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=2)(x1)

        # Second ResNetBlock (output filters = 64)
        x = ResNetBlock(in_channel=32, out_channel=64)(x)
        x2 = ResNetBlock(in_channel=64, out_channel=64)(x)
        x = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=2)(x2)

        # Third ResNetBlock (output filters = 128)
        x = ResNetBlock(in_channel=64, out_channel=128)(x)
        x3 = ResNetBlock(in_channel=128, out_channel=128)(x)
        x = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=2)(x3)

        # Fourth ResNetBlock (output filters = 256)
        x = ResNetBlock(in_channel=128, out_channel=256)(x)
        x = ResNetBlock(in_channel=256, out_channel=256)(x)
        x = ResNetBlock(in_channel=256, out_channel=256)(x)
        x4 = ResNetBlock(in_channel=256, out_channel=256)(x)

        return x1, x2, x3, x4