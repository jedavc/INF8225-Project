import torch.nn as nn
from math import sqrt

class ResidualPartD(nn.Module):
    def __init__(self, indim):
        super(ResidualPartD, self).__init__()
        self.conv1 = nn.Conv2d(indim, indim*2, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(indim*2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(indim*2, indim*2, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(indim*2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(indim*2, indim, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(indim)
        self.relu3 = nn.ReLU(inplace=True)
        #parameter initialization
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
        residual = self.conv1(x)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.relu2(residual)
        residual = self.conv3(residual)
        residual = self.relu3(residual)
        out = x + residual
        return out