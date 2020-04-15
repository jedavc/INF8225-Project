from architectures.DANet.utils.utils import prediction_to_segmentation
import torch
import os


def get_onehot_segmentation(target):
    batch_size, height, width = target.size()
    one_hot = torch.zeros(batch_size, 5, height, width, dtype=torch.float).cuda()

    return one_hot.scatter_(1, target.unsqueeze(1), 1.0)


def dice_score(pred, target):
    pred_onehot = prediction_to_segmentation(pred)
    target_onehot = get_onehot_segmentation(target)

    dims = (2, 3)
    intersection = torch.sum(pred_onehot * target_onehot, dims)
    cardinality = torch.sum(pred_onehot + target_onehot, dims)

    dice = (2. * intersection + 1e-8) / (cardinality + 1e-8)

    return torch.mean(dice, dim=0)


def dice_score_3d(path="../CHAOS_/val/result"):
    subjects_folder = os.listdir(path)
    for
