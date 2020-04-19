
import torch.nn as nn
from math import sqrt
import torch
import torch.nn.functional as F
from architectures.SegAN.model.ConvolutionalPart import ConvolutionalPart
from architectures.SegAN.model.ResidualPart import ResidualPart
from architectures.SegAN.model.ResidualPartD import ResidualPartD

channel_dim = 3
ndf = 64
class Segmentor(nn.Module):
    def __init__(self):
        super(Segmentor, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(channel_dim, ndf, 7, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convblock1_1 = ResidualPart(ndf)

        self.convblock2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convblock2_1 = ResidualPart(ndf * 2)

        self.convblock3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convblock3_1 = ResidualPart(ndf * 4)

        self.convblock4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convblock4_1 = ResidualPart(ndf * 8)

        self.convblock5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convblock5_1 = ResidualPart(ndf * 16)

        self.convblock6 = nn.Sequential(
            nn.Conv2d(ndf * 16, ndf * 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        #self.convblock6_1 = ResidualPart(ndf*32)


        self.deconvblock1 = nn.Sequential(
            # state size. (cat: ngf*32) x 1 x 1
            nn.Conv2d(ndf * 32, ndf * 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 2 x 2
        )
        self.deconvblock1_1 = ResidualPartD(ndf * 16)

        self.deconvblock2 = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            ConvolutionalPart(ndf * 16 * 2, ndf * 8, (7, 7)),
            # nn.ConvTranspose2d(ndf * 8 * 2, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
        )
        self.deconvblock2_1 = ResidualPartD(ndf * 8)

        self.deconvblock3 = nn.Sequential(
            # state size. (ngf*8) x 8 x 8
            ConvolutionalPart(ndf * 8 * 2, ndf * 4, (7, 7)),
            # nn.ConvTranspose2d(ndf * 8 * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
        )
        self.deconvblock3_1 = ResidualPartD(ndf * 4)
        self.deconvblock4 = nn.Sequential(
            # state size. (ngf*4) x 16 x 16
            ConvolutionalPart(ndf * 4 * 2, ndf * 2, (9, 9)),
            # nn.ConvTranspose2d(ndf * 4 * 2, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
        )
        self.deconvblock4_1 = ResidualPartD(ndf * 2)
        self.deconvblock5 = nn.Sequential(
            # state size. (ngf*2) x 32 x 32
            ConvolutionalPart(ndf * 2 * 2, ndf, (9, 9)),
            # nn.ConvTranspose2d(ndf * 2 * 2,     ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
        )
        self.deconvblock5_1 = ResidualPartD(ndf)
        self.deconvblock6 = nn.Sequential(
            # state size. (ngf) x 64 x 64
            ConvolutionalPart(ndf * 2, ndf, (11, 11)),
            # nn.ConvTranspose2d( ndf * 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
        )
        self.deconvblock6_1 = ResidualPartD(ndf)
        self.deconvblock7 = nn.Sequential(
            # state size. (ngf) x 128 x 128
            nn.Conv2d(ndf, 1, 5, 1, 2, bias=False),
            # state size. (channel_dim) x 128 x 128
            # nn.Sigmoid()
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        # for now it only supports one GPU
        encoder1 = self.convblock1(input)
        encoder1 = self.convblock1_1(encoder1)
        encoder2 = self.convblock2(encoder1)
        encoder2 = self.convblock2_1(encoder2)
        encoder3 = self.convblock3(encoder2)
        encoder3 = self.convblock3_1(encoder3)
        encoder4 = self.convblock4(encoder3)
        encoder4 = self.convblock4_1(encoder4) + encoder4
        encoder5 = self.convblock5(encoder4)
        encoder5 = self.convblock5_1(encoder5)
        encoder6 = self.convblock6(encoder5)
        #encoder6 = self.convblock6_1(encoder6)

        decoder1 = self.deconvblock1(encoder6)
        decoder1 = torch.cat([encoder5, decoder1], 1)
        decoder1 = F.upsample(decoder1, size=encoder4.size()[2:], mode='bilinear')
        decoder2 = self.deconvblock2(decoder1)
        decoder2 = self.deconvblock2_1(decoder2) + decoder2
        # concatenate along depth dimension
        decoder2 = torch.cat([encoder4, decoder2], 1)
        decoder2 = F.upsample(decoder2, size=encoder3.size()[2:], mode='bilinear')
        decoder3 = self.deconvblock3(decoder2)
        decoder3 = self.deconvblock3_1(decoder3)
        decoder3 = torch.cat([encoder3, decoder3], 1)
        decoder3 = F.upsample(decoder3, size=encoder2.size()[2:], mode='bilinear')
        decoder4 = self.deconvblock4(decoder3)
        decoder4 = self.deconvblock4_1(decoder4)
        decoder4 = torch.cat([encoder2, decoder4], 1)
        decoder4 = F.upsample(decoder4, size=encoder1.size()[2:], mode='bilinear')
        decoder5 = self.deconvblock5(decoder4)
        decoder5 = self.deconvblock5_1(decoder5)
        decoder5 = torch.cat([encoder1, decoder5], 1)
        decoder5 = F.upsample(decoder5, size = input.size()[2:], mode='bilinear')
        decoder6 = self.deconvblock6(decoder5)
        decoder6 = self.deconvblock6_1(decoder6)
        decoder7 = self.deconvblock7(decoder6)

        return decoder7