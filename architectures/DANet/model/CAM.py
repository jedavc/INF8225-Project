from torch import nn
from torchvision.models import resnext101_32x8d


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()