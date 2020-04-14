import torch
from torch import nn
import torch.nn.functional as F
import os
import torchvision


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def get_onehot_segmentation(target):
    batch_size, height, width = target.size()
    one_hot = torch.zeros(batch_size, 5, height, width, dtype=torch.float).cuda()

    return one_hot.scatter_(1, target.unsqueeze(1), 1.0)


# def dice_score(pred, target):
#     batch, num_class, height, width = pred.size()
#
#     num = pred * target
#     num = num.view(batch, num_class, -1).sum(dim=2)
#
#     den = pred.view(batch, num_class, -1).sum(dim=2)
#     den2 = target.view(batch, num_class, -1).sum(dim=2)
#
#     dice = (2 * num + 1e-8) / (den + den2 + 1e-8)
#
#     return dice.mean(dim=0)[1:]


def dice_score(pred, target):
    pred_soft = F.softmax(pred, dim=1)
    pred_onehot = predToSegmentation(pred_soft)
    target_onehot = get_onehot_segmentation(target)

    dims = (1, 2, 3)
    intersection = torch.sum(pred_onehot * target_onehot, dims)
    cardinality = torch.sum(pred_onehot + target_onehot, dims)

    dice = 2. * intersection / (cardinality + 1e-8)

    return torch.mean(dice)


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
