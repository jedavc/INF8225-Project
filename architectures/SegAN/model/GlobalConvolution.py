import torch.nn as nn
from math import sqrt


class GlobalConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1):
        super(GlobalConvolution, self).__init__()
        kernel_1 = kernel_size[0]
        kernel_2 = kernel_size[1]

        padding_1 = (kernel_1 - 1) // 2
        padding_2 = (kernel_2 - 1) // 2

        self.conv_1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_1, 1), padding=(padding_1, 0), stride=stride)
        self.conv_2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_2), padding=(0, padding_2), stride=stride)
        self.conv_3 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_2), padding=(0, padding_2), stride=stride)
        self.conv_4 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_1, 1), padding=(padding_1, 0), stride=stride)

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
        x_1 = self.conv_1(x)
        x_1 = self.conv_2(x_1)

        x_2 = self.conv_3(x)
        x_2 = self.conv_4(x_2)

        return x_1 + x_2
