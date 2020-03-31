from torch import nn


class Encoder(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(Encoder, self).__init__()

        