from torch import nn
from architectures.NVDLMED.model.ResNetBlock import *
from architectures.NVDLMED.model.DecoderGT import UpsampleBlock
from torch.distributions.normal import Normal


class VAE(nn.Module):
    def __init__(self, input_shape=(2, 160, 192, 128), output_channels=4):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.output_channels = output_channels

        self.vd_block = nn.Sequential(
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=16, kernel_size=(3, 3, 3), stride=2, padding=1),
            Flatten(),
            nn.Linear(1920, 256))

        self.vddraw_block = SamplingBlock(input_shape=256)

        self.vu_block = nn.Sequential(
            nn.Linear(128, (input_shape[0]//4) * (input_shape[1]//16) * (input_shape[2]//16) * (input_shape[3]//16)),
            nn.ReLU(),
            View((-1, input_shape[0]//4, input_shape[1]//16, input_shape[2]//16, input_shape[3]//16)),
            UpsampleBlock(in_channels=input_shape[0]//4, out_channels=256))

        self.vup_resnet_block2 = nn.Sequential(
            UpsampleBlock(in_channels=256, out_channels=128),
            ResNetBlock(in_channel=128))

        self.vup_resnet_block1 = nn.Sequential(
            UpsampleBlock(in_channels=128, out_channels=64),
            ResNetBlock(in_channel=64))

        self.vup_resnet_block0 = nn.Sequential(
            UpsampleBlock(in_channels=64, out_channels=32),
            ResNetBlock(in_channel=32))

        self.output_vae = nn.Conv3d(in_channels=32, out_channels=self.output_channels, kernel_size=(1, 1, 1), stride=1)

    def forward(self, x):
        # VD Block (Reducing dimensionality of the data)
        x = self.vd_block(x)

        # Sampling BLock
        x, mu, logvar = self.vddraw_block(x)

        # VU BLock (Upsizing back to a depth of 256)
        x = self.vu_block(x)

        # First Decoder ResNetBlock (output filters = 128)
        x = self.vup_resnet_block2(x)

        # Second Decoder ResNetBlock (output filters = 64)
        x = self.vup_resnet_block1(x)

        # Third Decoder ResNetBlock (output filters = 32)
        x = self.vup_resnet_block0(x)

        # Output Block
        out_vae = self.output_vae(x)

        return out_vae, mu, logvar


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        out = x.view(batch_size, -1)

        return out


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class SamplingBlock(nn.Module):
    def __init__(self, input_shape=256):
        super(SamplingBlock, self).__init__()

        self.input_shape = input_shape

    @staticmethod
    def sampling(z_mean, z_var):
        epsilon = torch.rand_like(z_mean).cuda()

        return z_mean + torch.exp(0.5 * z_var) * epsilon

    def forward(self, x):
        z_mean = x[:, :(self.input_shape // 2)]
        z_var = x[:, (self.input_shape // 2):]

        out = self.sampling(z_mean, z_var)

        return out, z_mean, z_var
