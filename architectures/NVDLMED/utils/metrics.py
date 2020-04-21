import numpy as np
from medpy.metric.binary import *


def calculate_3d_metrics(pred, gt):
    batch, c, H, W, D = pred.size()

    dsc_3d = np.zeros((batch, c))
    hd_3d = np.zeros((batch, c))
    for subj_i in range(batch):
        for tumor_class in range(c):
            # Convertir les pred en 1-hot
            single_class_pred = pred[subj_i, tumor_class].cpu().numpy()
            single_class_gt = gt[subj_i, tumor_class].cpu().numpy()

            dsc_3d[subj_i, tumor_class] = dc(single_class_pred, single_class_gt)
            hd_3d[subj_i, tumor_class] = hd(single_class_pred, single_class_gt)

    return dsc_3d, hd_3d


def convert_to_bool(tensor):
    pass