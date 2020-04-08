from .PreAttentionModule import *
from .ResNextModule import *
from .InterpolationModule import InterpolationModule


class MSDualGuided(nn.Module):
    def __init__(self):
        super(MSDualGuided, self).__init__()

        self.resnext_module = ResNextModule()
        self.preattention_module = PreAttentionModule()

    def forward(self, x):
        f1, f2, f3, f4 = self.resnext_module(x)
        fsp, fms, fsms, predict1 = self.preattention_module(f1, f2, f3, f4)
