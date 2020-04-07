from torch import nn
import torch


class CAM(nn.Module):
    def __init__(self, input_channels):
        super(CAM, self).__init__()

        self.input_channels = input_channels

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, C, H, W = x.size()

        proj_query = x.view(batch, C, -1)
        proj_key = x.view(batch, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out