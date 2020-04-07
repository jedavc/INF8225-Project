from torch import nn
from torchvision.models import resnext101_32x8d


class ResNextBlock(nn.Module):
    def __init__(self):
        super(ResNextBlock, self).__init__()

        self.resnext = self.reformated_resnext()

        self.layer1 = nn.Sequential(*self.resnext[0:5])
        self.layer2 = self.resnext[5]
        self.layer3 = self.resnext[6]
        self.layer4 = self.resnext[7]

    @staticmethod
    def reformated_resnext():
        resnext = resnext101_32x8d()

        resnext.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7),
                                       stride=(2, 2), padding=(3, 3), bias=False)

        # resnext.avgpool = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
        #
        # resnext.fc = nn.Sequential(
        #     Lambda(lambda x: x.view(x.size(0), -1)),
        #     Lambda(lambda x: x.view(x.size(0), -1)),
        #     nn.Linear(2048, 1000)
        # )

        return list(resnext.children())

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        return layer1, layer2, layer3, layer4


# class Lambda(nn.Module):
#     def __init__(self, fn):
#         super(Lambda, self).__init__()
#         self.lambda_func = fn
#
#     def forward(self, input):
#         return self.lambda_func
#
