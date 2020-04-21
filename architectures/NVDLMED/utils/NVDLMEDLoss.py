from torch import nn
import torch


class NVDLMEDLoss(nn.Module):
    def __init__(self):
        super(NVDLMEDLoss, self).__init__()
        self.loss_l2 = nn.MSELoss()

    def forward(self, output_gt, gt, output_vae, input, mu, var):
        l_dice = loss_dice(output_gt, gt)
        l_l2 = self.loss_l2(output_vae, input)
        l_kl = loss_kl(mu, var, input[0].size())

        return l_dice + 0.1 * l_l2 + 0.1 * l_kl, l_dice, l_l2, l_kl


def loss_dice(pred, gt, epsilon=1e-6):
    normalized_pred = torch.sigmoid(pred)

    normalized_pred = flatten(normalized_pred)
    gt = flatten(gt).float()

    intersect = (normalized_pred * gt).sum(-1)
    denominator = (normalized_pred * normalized_pred).sum(-1) + (gt * gt).sum(-1)

    per_channel_dice = 2 * (intersect / denominator.clamp(min=epsilon))

    return (1. - torch.mean(per_channel_dice))


def loss_kl(z_mean, z_var, input_shape):
    c, H, W, D = input_shape
    n = c * H * W * D

    return torch.sum(z_var.exp() + z_mean.pow(2) - 1. - z_var) #* 0.5


def flatten(tensor):
    # number of channels
    C = tensor.size(1)

    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))

    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)

    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)