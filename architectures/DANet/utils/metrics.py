import torch
from torch import nn
import os
import torchvision

def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def getOneHotSegmentation(batch):
    backgroundVal = 0

    # Chaos MRI (These values are to set label values as 0,1,2,3 and 4)
    label1 = 0.24705882
    label2 = 0.49411765
    label3 = 0.7411765
    label4 = 0.9882353

    oneHotLabels = torch.cat(
        (batch == backgroundVal, batch == label1, batch == label2, batch == label3, batch == label4),
        dim=1)

    return oneHotLabels.float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.3137255, 0.627451 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3

    denom = 0.24705882  # for Chaos MRI  Dataset this value

    return (batch / denom).round().long().squeeze(dim=1)


def dice_score(pred, target):
    batch, num_class, height, width = target.size()

    num = pred * target
    num = num.view(batch, num_class, -1).sum(dim=2)

    den = pred.view(batch, num_class, -1).sum(dim=2)
    den2 = target.view(batch, num_class, -1).sum(dim=2)

    dice = (2 * num + 1e-8) / (den + den2 + 1e-8)

    return dice.mean(dim=0)[1:]


def dice_comme_eux(pred, target):
    num = pred * target
    num = num.sum(dim=3).sum(dim=2).sum(dim=0)

    den = pred.sum(dim=3).sum(dim=2).sum(dim=0)
    den2 = target.sum(dim=3).sum(dim=2).sum(dim=0)

    dice = (2 * num + 1e-8) / (den + den2 + 1e-8)

    return dice[1:]


def getSingleImage(pred):
    # input is a 4-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    num_classes = 5
    Val = torch.zeros(num_classes).cuda()

    # Chaos MRI
    Val[1] = 0.24705882
    Val[2] = 0.49411765
    Val[3] = 0.7411765
    Val[4] = 0.9882353

    x = predToSegmentation(pred)

    out = x * Val.view(1, 5, 1, 1)

    return out.sum(dim=1, keepdim=True)


def saveImages_for3D(pred, img_path):
    path = '../Results/DANet/val'
    if not os.path.exists(path):
        os.makedirs(path)


    segmentation = getSingleImage(pred)

    str_1 = img_path[0].replace("\\", "/").split('/Img/')
    str_subj = str_1[1].replace("\\", "/").split('slice')

    path_Subj = path + '/' + str_subj[0]
    if not os.path.exists(path_Subj):
        os.makedirs(path_Subj)

    str_subj = str_subj[1].split('_')
    torchvision.utils.save_image(segmentation.data, os.path.join(path_Subj, str_subj[1]))