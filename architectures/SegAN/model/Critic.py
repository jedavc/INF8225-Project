import torch.nn as nn
from math import sqrt
import torch
from architectures.SegAN.model.GlobalConvolution import GlobalConvolution

channel_dim = 3
ndf = 64
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.conv1 = nn.Sequential(
            GlobalConvolution(channel_dim, ndf, (13, 13), 2),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            GlobalConvolution(ndf, ndf * 2, (11, 11), 2),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            GlobalConvolution(ndf * 2, ndf * 4, (9, 9), 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )


        self.conv4 = nn.Sequential(
            GlobalConvolution(ndf * 4, ndf * 8, (7, 7), 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, 3, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # self._initialize_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, input):
        batch_size = input.size()[0]
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)
        output = torch.cat((input.view(batch_size, -1), 1 * out1.view(batch_size, -1),
                            2 * out2.view(batch_size, -1), 2 * out3.view(batch_size, -1),
                            2 * out4.view(batch_size, -1), 2 * out5.view(batch_size, -1),
                            4 * out6.view(batch_size, -1)), 1)
        return output