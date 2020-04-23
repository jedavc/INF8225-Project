import nibabel as nib
import numpy as np
import os
from architectures.NVDLMED.utils.constant import *
import torch


def prediction_to_nii(pred, gt, img_names, out_path):
    pred_one_hot = (pred > 0.5).float()
    pred_seg = channels_to_segmentation(pred_one_hot)
    gt_seg = channels_to_segmentation(gt)


    for i, name in enumerate(img_names):
        patient_pred = pred_seg[i].cpu().numpy()
        patient_gt = gt_seg[i].cpu().numpy()

        nii_pred = nib.Nifti1Image(patient_pred, affine=np.eye(4))
        nii_gt = nib.Nifti1Image(patient_gt, affine=np.eye(4))

        nib.save(nii_pred, os.path.join(out_path, name))
        nib.save(nii_gt, os.path.join(out_path, name + "_gt"))


def channels_to_segmentation(one_hot_tensor):
    brats_pixel_values = np.array([MASK_NET, MASK_ED, MASK_ET])
    brats_pixel = torch.from_numpy(brats_pixel_values).cuda()

    out = one_hot_tensor * brats_pixel.view(1, 3, 1, 1, 1)

    return out.sum(dim=1)