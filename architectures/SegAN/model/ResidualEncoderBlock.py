import torch.nn as nn
from math import sqrt


class ResidualEncoderBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(dim * 2)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(dim * 2)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(dim)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        # init params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        output = self.conv1(x)
        output = self.act1(output)
        output = self.batch_norm1(output)
        output = self.conv2(output)
        output = self.act2(output)
        output = self.batch_norm2(output)
        output = self.conv3(output)
        output = self.act3(output)
        output = self.batch_norm3(output)

        return x + output
