from torch import nn
import torch


class PAM(nn.Module):
    def __init__(self, in_dim):
        super(PAM, self).__init__()

        self.in_dim = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, C, H, W = x.size()

        proj_query = self.query_conv(x).view(batch, -1, W * H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, W * H)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch, -1, W * H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)
        out = self.gamma * out + x

        return out


