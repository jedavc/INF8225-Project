from architectures.SegAN.model import *
from datasets import MelanomaDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import torch.nn.functional as F
import argparse
import logging

parser.add_argument('--cuda', action='store_true', help='using GPU or not')

# Segmentor -> Generator
# Critic - > Discriminator
output_path = "./outputs"
data_path = "./data"
num_epoch = 100
lr = 0.003
decay = 0.5
batch_size = 25
S_losses = []
C_losses = []

def dice_loss(input,target):
    num = input * target
    num = torch.sum(num,dim=2)
    num = torch.sum(num,dim=2)

    den1 = input * input
    den1 = torch.sum(den1,dim=2)
    den1 = torch.sum(den1,dim=2)

    den2 = target * target
    den2 = torch.sum(den2,dim=2)
    den2 = torch.sum(den2,dim=2)

    dice = 2 * (num/(den1+den2))

    dice_total = 1 - 1 * torch.sum(dice)/dice.size(0)#divide by batchsize

    return dice_total

def update_optimizer(lr, decay, Segmentor_params, Critic_params):
    optimizer_seg = Adam(Segmentor_params,lr=lr, betas=(0.5, 0.999))
    optimizer_crit = Adam(Segmentor_params,lr=lr, betas=(0.5, 0.999))
    return optimizer_seg, optimizer_crit

def loadData(set, path,)
if __name__ == "__main__":
    mel
    ##torch.manual_seed(opt.seed) Set la seed
    cuda = opt.cuda

    logging.info("Build Model")

    Segmentor = Segmentor()
    logging.debug(Segmentor)

    Critic = Critic()
    logging.debug(Critic)

    if cuda:
        Segmentor.to(‘cuda’)
        Critic.to(‘cuda’)

    optimizer_seg, optimizer_crit = update_optimizer(lr,decay,Segmentor.parameters(), Critic.parameters())

    Segmentor.train()

    train_loader = loader(MelanomaDataset.Dataset(data_path), batch_size)
    val_loader = loader(MelanomaDataset.Dataset_val(data_path), batch_size)
    for epoch in range(num_epoch):
        for (input, target) in train_loader:
            input, target = input.to(‘cuda’), target.to(‘cuda’)
            #
            output = Segmentor(input)
            output = F.sigmoid(output).detach()
            output_masked = input.clone()
            input_mask = intput.clone()

            # detach G from the network
            for d in range(3):
                output_masked[:,d,:,:] = (input_mask[:,d,:,:].unsqueeze(1) * output).squeeze()

            output_masked = output_masked.to(‘cuda’)
            
            result = Critic(output_masked)
            target_masked = input.clone()

            for d in range(3):
                target_masked[:,d,:,:] = (input_mask[:,d,:,:].unsqueeze(1) * target).squeeze()
            target_masked = target_masked.cuda()

            target_C = Critic(target_masked)
            loss_C = - torch.mean(torch.abs(result - target_C))
            loss_C.backward()
            optimizer_crit.step()

            #clip parameters in D
            for p in Critic.parameters():
                p.data.clamp_(-0.05, 0.05)
            #train G
            Segmentor.zero_grad()
            output = Segmentor(input)
            output = F.sigmoid(output)
    
            for d in range(3):
                output_masked[:,d,:,:] = (input_mask[:,d,:,:].unsqueeze(1) * output).squeeze()
            
            output_masked = output_masked.cuda()
            result = Critic(output_masked)
            for d in range(3):
                target_masked[:,d,:,:] = (input_mask[:,d,:,:].unsqueeze(1) * target).squeeze()
            if cuda:
                target_masked = target_masked.cuda()
            target_S = Critic(target_masked)
            loss_dice = dice_loss(output,target)
            print("Loss dice: " + str(loss_dice.item()))
            loss_S = torch.mean(torch.abs(result - target_S))
            loss_S_joint = torch.mean(torch.abs(result - target_S)) + loss_dice
            loss_S_joint.backward()
            optimizer_seg.step()

        print("===> Epoch[{}]({}/{}): Batch Dice: {:.4f}".format(epoch, i, len(dataloader), 1 - loss_dice.item()))
        print("===> Epoch[{}]({}/{}): G_Loss: {:.4f}".format(epoch, i, len(dataloader), loss_G.item()))
        print("===> Epoch[{}]({}/{}): D_Loss: {:.4f}".format(epoch, i, len(dataloader), loss_D.item()))
        
        if epoch % 10 == 0:
            Segmentor.eval()
            IoUs, dices = [], []
            for (input, gt) in val_loader:
                input, gt = input.cuda(), gt.cuda() 

                pred = Segmentor(input)
                pred[pred < 0.5] = 0
                pred[pred >= 0.5] = 1
                pred = pred.type(torch.LongTensor)
                pred_np = pred.data.cpu().numpy()
                gt = gt.data.cpu().numpy()
                for x in range(input.size()[0]):
                    IoU = np.sum(pred_np[x][gt[x]==1]) / float(np.sum(pred_np[x]) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x]==1]))
                    dice = np.sum(pred_np[x][gt[x]==1])*2 / float(np.sum(pred_np[x]) + np.sum(gt[x]))
                    IoUs.append(IoU)
                    dices.append(dice)

            Segmentor.train()
            IoUs = np.array(IoUs, dtype=np.float64)
            dices = np.array(dices, dtype=np.float64)
            mIoU = np.mean(IoUs, axis=0)
            mdice = np.mean(dices, axis=0)
            print('mIoU: {:.4f}'.format(mIoU))
            print('Dice: {:.4f}'.format(mdice))

        # Adjust epoch number
        if epoch % 25 == 0:
                    lr = lr*decay
                    if lr <= 0.00000001:
                        lr = 0.00000001
                    print('Learning Rate: {:.6f}'.format(lr))
                    # print('K: {:.4f}'.format(k))
                    print('Max mIoU: {:.4f}'.format(max_iou))
                    optimizer_seg, optimizer_crit = update_optimizer(lr,decay,Segmentor.parameters(), Critic.parameters())


