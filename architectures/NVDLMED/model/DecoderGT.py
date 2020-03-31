from torch import nn


class DecoderGT(nn.Module):
    def __init__(self):
        super(DecoderGT, self).__init__()

    def forward(self, x1, x2, x3, x4):

        # First ResNetBlock (output filters = 128)
        x = nn.Conv3d(in_channels=x4.shape[1], out_channels=128, kernel_size=(1, 1, 1), stride=1)(x4)
        x = nn.Upsample(size=2*x.shape)