from torch import nn


class MSDualGuidedLoss(nn.Module):
    def __init__(self):
        super(MSDualGuidedLoss, self).__init__()

        self.softmax = nn.Softmax()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2, mask):
        segmentation_class = getTargetSegmentation(mask)

        predict_loss = sum([self.ce_loss(predict1[i], segmentation_class) + self.ce_loss(predict2[i], segmentation_class) for i in range(len(predict1))])
        sementic_loss = sum([self.mse_loss(semVector1[i], semVector2[i]) for i in range(len(semVector1))])
        reconst_loss = sum([self.mse_loss(fsms[i], semModule1[i]) + self.mse_loss(fai[i], semModule2[i]) for i in range(len(semModule1))])

        total_loss = predict_loss + 0.25 * sementic_loss + 0.1 * reconst_loss

        return total_loss


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.3137255, 0.627451 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3

    denom = 0.24705882  # for Chaos MRI  Dataset this value

    return (batch / denom).round().long().squeeze(dim=1)
