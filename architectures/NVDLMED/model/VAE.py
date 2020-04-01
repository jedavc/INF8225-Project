from torch import nn
from architectures.NVDLMED.model.ResNetBlock import *
from torch.distributions.normal import Normal


class VAE(nn.Module):
    def __init__(self, output_channels=3):
        super(VAE, self).__init__()
        self.output_channels = output_channels

        self.activation = nn.ReLU()

    def forward(self, x):
        # VD Block
        x = GroupNorm(x.size(1), num_groups=8)(x)
        x = self.activation(x)
        x = nn.Conv3d(in_channels=x.size(1), out_channels=16, kernel_size=(3, 3, 3), stride=2)(x)

        # Sampling BLock
        x = SamplingBlock(input_shape=256, output_shape=128)(x)

        # VU BLock
        


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
