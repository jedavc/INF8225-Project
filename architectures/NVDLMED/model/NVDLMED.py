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
        encoded = self.encoder(x)
        decoded_gt = self.decoder_gt(encoded)
        decoded_vae = self.vae(encoded)

        return decoded_gt, decoded_vae

print(list(NVDLMED().parameters()))