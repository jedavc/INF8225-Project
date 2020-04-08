from keras.layers import *

class PairBlock():
    def __init__(self, out_channels):
        self.out_channels = out_channels
        self.conv1 = Conv2D(self.out_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(self.out_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')

    def __call__(self, input_values):
        conv1 = self.conv1(input_values)
        conv2 = self.conv2(conv1)
        return conv2




#
# input = Input((256,256,1))
# test1 = PairBlock(12)(input)
# test2 = PairBlock(20)
#
# in1 = test1(input)
# in2 = test2(in1)
#
# model = Model(input = input, output=in2)
# print(model.summary())