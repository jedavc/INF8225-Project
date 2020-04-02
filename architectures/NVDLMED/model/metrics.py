import numpy as np
import torch


def dice_coefficient(y_true, y_pred):
    im1 = np.asarray(y_true).astype(np.bool)
    im2 = np.asarray(y_pred).astype(np.bool)

    intersection = np.logical_and(y_true, y_pred)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def loss_gt(gt, pred):
    iflat = pred.contiguous().view(-1)
    tflat = gt.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = (tflat * iflat).sum()
    B_sum = (tflat * tflat).sum()

    return 1 - ((2. * intersection) / (A_sum + B_sum + 1e-8))