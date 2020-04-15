import torch
import os
import torch.nn.functional as F
import torchvision
from architectures.DANet.utils.constant import *
import numpy as np


def prediction_to_segmentation(pred):
    soft_pred = F.softmax(pred)

    Max = soft_pred.max(dim=1, keepdim=True)[0]
    x = soft_pred / Max

    return (x == 1).float()


def prediction_to_normalized_pil(pred_onehot):
    chaos_pixel_values = np.array([MASK_BG, MASK_LIVER, MASK_KR, MASK_KL, MASK_SPLEEN])
    chaos_pixel = torch.from_numpy(chaos_pixel_values).cuda()

    out = pred_onehot * chaos_pixel.view(1, 5, 1, 1)

    return out.sum(dim=1, keepdim=True)


def prediction_to_png(pred, img_path, out_path='../rawdata/CHAOS_/val/result'):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    pred_onehot = prediction_to_segmentation(pred)
    normalized_pil = prediction_to_normalized_pil(pred_onehot)

    file_name = os.path.abspath(img_path[0]).split("\\")[-1]
    torchvision.utils.save_image(normalized_pil.data, os.path.join(out_path, file_name))
