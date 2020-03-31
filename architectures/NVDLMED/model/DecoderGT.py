from torch import nn


class DecoderGT(nn.Module):
    def __init__(self, input_shape=(4, 160, 192, 128)):
        super(DecoderGT, self).__init__()

        self.input_shape = input_shape

    def forward(self, x):
        