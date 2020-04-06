from torch import nn
from architectures.NVDLMED.model.Encoder import *
from architectures.NVDLMED.model.DecoderGT import *
from architectures.NVDLMED.model.VAE import *


class NVDLMED(nn.Module):
    def __init__(self, input_shape=(4, 160, 192, 128), output_gt=3, output_vae=4):
        super(NVDLMED, self).__init__()

        self.input_shape = input_shape
        self.encoder = Encoder(input_shape=input_shape)
        self.decoder_gt = DecoderGT(output_channels=output_gt)
        self.vae = VAE(input_shape=input_shape, output_channels=output_vae)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        decoded_gt = self.decoder_gt(x1, x2, x3, x4)
        decoded_vae = self.vae(x4)

        return decoded_gt, decoded_vae