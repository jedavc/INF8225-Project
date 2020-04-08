from architectures.BCDU_net.model.PairBlock import *


class MaxPoolConvBlock():
    def __init__(self, out_channels):
        self.out_channels = out_channels
        self.maxPool1 = MaxPooling2D(pool_size=(2, 2))
        self.pairBlock1 = PairBlock(self.out_channels)


    def __call__(self, input_values):
        maxPool1 = self.maxPool1(input_values)
        pairBlock1 = self.pairBlock1(maxPool1)
        return pairBlock1


# input = Input((256,256,1))
# test1 = MaxPoolConvBlock(12)
# test2 = MaxPoolConvBlock(20)
#
# in1 = test1(input)
# in2 = test2(in1)
#
# model = Model(input = input, output=in2)
# print(model.summary())