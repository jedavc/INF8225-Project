from torch import nn
from architectures.NVDLMED.model.ResNetBlock import *
from torch.distributions.normal import Normal


class VAE(nn.Module):
    def __init__(self, input_shape=(2, 160, 192, 128), output_channels=4):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.output_channels = output_channels

        self.vd_block = nn.Sequential(nn.GroupNorm(8, 256),
                                      nn.ReLU(),
                                      nn.Conv3d(in_channels=256, out_channels=16, kernel_size=(3, 3, 3), stride=2, padding=1),
                                      Flatten(),
                                      nn.Linear(3360, 256))

        self.vddraw_block = SamplingBlock(input_shape=256, output_shape=128)

        self.vu_block = nn.Sequential(nn.Linear(128, (input_shape[0]//4) * (input_shape[1]//16) * (input_shape[2]//16) * (input_shape[3]//16)),
                                      nn.ReLU(),
                                      View((-1, input_shape[0]//4, input_shape[1]//16, input_shape[2]//16, input_shape[3]//16)),
                                      nn.Conv3d(in_channels=input_shape[0]//4, out_channels=256, kernel_size=(1, 1, 1), stride=1),
                                      nn.Upsample(scale_factor=2, mode='trilinear'))

        self.vup_resnet_block2 = nn.Sequential(nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(1, 1, 1), stride=1),
                                              nn.Upsample(scale_factor=2, mode='trilinear'),
                                              ResNetBlock(in_channel=128, out_channel=128))

        self.vup_resnet_block1 = nn.Sequential(nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(1, 1, 1), stride=1),
                                               nn.Upsample(scale_factor=2, mode='trilinear'),
                                               ResNetBlock(in_channel=64, out_channel=64))

        self.vup_resnet_block0 = nn.Sequential(nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(1, 1, 1), stride=1),
                                               nn.Upsample(scale_factor=2, mode='trilinear'),
                                               ResNetBlock(in_channel=32, out_channel=32))

        self.output_vae = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=self.output_channels, kernel_size=(1, 1, 1), stride=1))

    def forward(self, x):
        # VD Block (Reducing dimensionality of the data)
        x = self.vd_block(x)

        # Sampling BLock
        x = self.vddraw_block(x)

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

        return out_vae


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
    def __init__(self, input_shape=256, output_shape=128):
        super(SamplingBlock, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.dense_mean = nn.Linear(256, self.output_shape)
        self.dense_var = nn.Linear(256, self.output_shape)

    def sampling(self, z_mean, z_var):
        batch, dim = z_mean.size()
        epsilon = torch.randn(size=(batch, dim)).cuda()

        return z_mean + torch.exp(0.5 * z_var) * epsilon

    def forward(self, x):
        # VDraw Block (Sampling)
        z_mean = self.dense_mean(x)
        z_var = self.dense_mean(x)
        out = self.sampling(z_mean, z_var)

        return out
