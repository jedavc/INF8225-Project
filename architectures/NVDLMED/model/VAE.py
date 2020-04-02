from torch import nn
from architectures.NVDLMED.model.ResNetBlock import *
from torch.distributions.normal import Normal


class VAE(nn.Module):
    def __init__(self, input_shape=(4, 160, 192, 128), output_channels=3):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.output_channels = output_channels

        self.activation = nn.ReLU()
        self.upsample3d = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        # VD Block (Reducing dimensionality of the data)
        x = GroupNorm(x.size(1), num_groups=8)(x)
        x = self.activation(x)
        x = nn.Conv3d(in_channels=x.size(1), out_channels=16, kernel_size=(3, 3, 3), stride=2)(x)

        # Sampling BLock
        x = SamplingBlock(input_shape=256, output_shape=128)(x)

        # VU BLock (Upsizing back to a depth of 256)
        c, H, W, D = self.input_shape
        x = nn.Linear(x.size(1), (c//4) * (H//16) * (W//16) * (D//16))(x)
        x = self.activation(x)
        x = x.view(-1, (c//4), (H//16), (W//16), (D//16))
        x = nn.Conv3d(in_channels=x.size(1), out_channels=256, kernel_size=(1, 1, 1), stride=1)(x)
        x = self.upsample3d(x)

        # First Decoder ResNetBlock (output filters = 128)
        x = nn.Conv3d(in_channels=x.size(1), out_channels=128, kernel_size=(1, 1, 1), stride=1)(x)
        x = self.upsample3d(x)
        x = ResNetBlock(in_channel=128, out_channel=128)(x)

        # Second Decoder ResNetBlock (output filters = 64)
        x = nn.Conv3d(in_channels=x.size(1), out_channels=64, kernel_size=(1, 1, 1), stride=1)(x)
        x = self.upsample3d(x)
        x = ResNetBlock(in_channel=64, out_channel=64)(x)

        # Third Decoder ResNetBlock (output filters = 32)
        x = nn.Conv3d(in_channels=x.size(1), out_channels=32, kernel_size=(1, 1, 1), stride=1)(x)
        x = self.upsample3d(x)
        x = ResNetBlock(in_channel=32, out_channel=32)(x)

        # Blue Block (output filters=32)
        x = nn.Conv3d(in_channels=x.size(1), out_channels=32, kernel_size=(3, 3, 3), stride=1)(x)

        # Output Block
        out_VAE = nn.Conv3d(in_channels=x.size(1), out_channels=self.output_channels, kernel_size=(1, 1, 1), stride=1)(x)

        return out_VAE


class SamplingBlock(nn.Module):
    def __init__(self, input_shape=256, output_shape=128):
        super(SamplingBlock, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

    def sampling(self, z_mean, z_var):
        batch, dim = z_mean.size()
        epsilon = torch.randn(size=(batch, dim))

        return z_mean + torch.exp(0.5 * z_var) * epsilon

    def forward(self, x):
        # Flattening layer
        x = x.view(x.size(0), self.input_shape)
        x = nn.Linear(x.size(1), 256)(x)

        # VDraw Block (Sampling)
        z_mean = nn.Linear(self.input_shape, self.output_shape)(x)
        z_var = nn.Linear(self.input_shape, self.output_shape)(x)
        out = self.sampling(z_mean, z_var)

        return out
