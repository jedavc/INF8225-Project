from architectures.SegAN.model.Segmentor import *
from architectures.SegAN.model.Critic import *
from datasets.MelanomaDataset import *
import torch
import torchvision.utils as vutils
from torch.optim import Adam
import torch.nn.functional as F
import logging
import os
import matplotlib.pyplot as plt

# Segmentor -> Generator
# Critic - > Discriminator
output_path = "./outputs"
data_path = "./datasets/data"
num_epoch = 200
lr = 0.003
lr_decay = 0.5
batch_size = 12
k_decay = 0.9
k = 1

def dice_loss(input,target):
    num = input * target
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=2)

    den1 = input * input
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=2)

    den2 = target * target
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=2)

    dice = 2 * (num/(den1+den2))

    dice_total = 1 - 1 * torch.sum(dice)/dice.size(0)#divide by batchsize

    return dice_total

def update_optimizer(lr, Segmentor_params, Critic_params):
    optimizer_seg = Adam(Segmentor_params, lr=lr, betas=(0.5, 0.999))
    optimizer_crit = Adam(Critic_params, lr=lr, betas=(0.5, 0.999))
    return optimizer_seg, optimizer_crit

def get_accuracy(model, data_loader, output_path):
    with torch.no_grad():
        correct = 0
        for data, label in data_loader:
            data, label = data.cuda(), label.cuda()
            predictions = model(data)
            predictions[predictions < 0.4] = 0
            predictions[predictions >= 0.4] = 1
            _, predicted = torch.max(predictions, 1)
            correct += (predicted == label).sum().item()/label.nelement()

        if epoch % 10 == 0:
            vutils.save_image(data.double(),
                              '%s/input_val.png' % output_path,
                              normalize=True)
            vutils.save_image(target.double(),
                              '%s/label_val.png' % output_path,
                              normalize=True)
            vutils.save_image(predictions.double(),
                              '%s/result_val.png' % output_path,
                              normalize=True)
    return correct / len(data_loader.dataset)

if __name__ == "__main__":


    cwd = os.getcwd()
    print(cwd)
    ##torch.manual_seed(opt.seed) Set la seed
    cuda = True

    logging.info("Build Model")

    Segmentor = Segmentor()
    logging.debug(Segmentor)

    Critic = Critic()
    logging.debug(Critic)

    if cuda:
        Segmentor.cuda()
        Critic.cuda()

    optimizer_seg, optimizer_crit = update_optimizer(lr, Segmentor.parameters(), Critic.parameters())

    max_iou = 0
    Segmentor.train()
    Critic.train()
    train_loader = loader(Dataset(data_path), batch_size)
    val_loader = loader(Dataset_val(data_path), batch_size)
    losses_S = []
    losses_C = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(num_epoch):
        correct_train = 0
        for batch_idx, sample in enumerate(train_loader):

            Critic.zero_grad()
            input, target = sample[0].cuda(), sample[1].cuda()
            output = Segmentor(input)
            output = F.sigmoid(output*k)
            output = output.detach()
            output_masked = input.clone()
            input_mask = input.clone()

            # detach G from the network
            for d in range(3):
                output_masked[:,d,:,:] = (input_mask[:, d,:,:].unsqueeze(1) * output).squeeze()

            output_masked = output_masked.cuda()
            
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
            output = F.sigmoid(output*k)
    
            for d in range(3):
                output_masked[:,d,:,:] = (input_mask[:,d,:,:].unsqueeze(1) * output).squeeze()
            
            output_masked = output_masked.cuda()
            result = Critic(output_masked)
            for d in range(3):
                target_masked[:, d, :, :] = (input_mask[:,d,:,:].unsqueeze(1) * target).squeeze()

            target_masked = target_masked.cuda()
            target_S = Critic(target_masked)
            loss_dice = dice_loss(output, target)
            print("Loss dice: " + str(loss_dice.item()))
            loss_S = torch.mean(torch.abs(result - target_S))
            loss_S_joint = torch.mean(torch.abs(result - target_S)) + loss_dice
            loss_S_joint.backward()
            optimizer_seg.step()

            # Accuracy
            output[output < 0.4] = 0
            output[output >= 0.4] = 1
            _, binary_output = torch.max(output, 1)
            correct_train += (binary_output == target).sum().item()/target.nelement()

        train_accuracy = correct_train / len(train_loader.dataset)
        print("Train_accuracy : {}".format(train_accuracy))

        train_accuracies.append(correct_train)
        losses_C.append(loss_C.item())
        losses_S.append(loss_S.item())


        print("===> Epoch[{}]({}/{}): Batch Dice: {:.4f}".format(epoch, batch_idx, len(train_loader), 1 - loss_dice.item()))
        print("===> Epoch[{}]({}/{}): Segmentor_Loss: {:.4f}".format(epoch, batch_idx, len(train_loader), loss_S.item()))
        print("===> Epoch[{}]({}/{}): Critic_Loss: {:.4f}".format(epoch, batch_idx, len(train_loader), loss_C.item()))
        vutils.save_image(input.double(),
                '%s/input.png' % output_path,
                normalize=True)
        vutils.save_image(target.double(),
                '%s/label.png' % output_path,
                normalize=True)
        vutils.save_image(output.double(),
                '%s/result.png' % output_path,
                normalize=True)

        Segmentor.eval()
        # Val Accuracy
        val_accuracy = get_accuracy(Segmentor, val_loader, output_path)
        val_accuracies.append(val_accuracy)
        print("Val_accuracy : {}".format(val_accuracy))

        Segmentor.train()

        """  if epoch % 10 == 0:
                correct = 0
                Segmentor.eval()
                IoUs, dices = [], []
                for batch_idx, sample in enumerate(val_loader):
                    input, gt = sample[0].cuda(), sample[1].cuda()
    
                    pred = Segmentor(input)
                    pred[pred < 0.4] = 0
                    pred[pred >= 0.4] = 1
                    pred = pred.type(torch.LongTensor)
                    pred_np = pred.data.cpu().numpy()
                    gt = gt.data.cpu().numpy()
                    for x in range(input.size()[0]):
                        IoU = np.sum(pred_np[x][gt[x] == 1]) / float(np.sum(pred_np[x]) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x] == 1]))
                        dice = np.sum(pred_np[x][gt[x] == 1]) * 2 / float(np.sum(pred_np[x]) + np.sum(gt[x]))
                        IoUs.append(IoU)
                        dices.append(dice)
    
                accuracy = 100 * correct / len(val_loader.dataset)
                val_accuracies.append(accuracy)
                Segmentor.train()
                IoUs = np.array(IoUs, dtype=np.float64)
                dices = np.array(dices, dtype=np.float64)
                mIoU = np.mean(IoUs, axis=0)
                mdice = np.mean(dices, axis=0)
                print('mIoU: {:.4f}'.format(mIoU))
                print('Dice: {:.4f}'.format(mdice))
                if mIoU > max_iou:
                    max_iou = mIoU
                torch.save(Segmentor.state_dict(), '%s/NetS_epoch_%d.pth' % (output_path, epoch))
                vutils.save_image(sample[0].double(),
                                  '%s/input_val.png' % output_path,
                                  normalize=True)
                vutils.save_image(sample[1].double(),
                                  '%s/label_val.png' % output_path,
                                  normalize=True)
                pred = pred.type(torch.FloatTensor)
                vutils.save_image(pred.data.double(),
                                  '%s/result_val.png' % output_path,
                                  normalize=True) """

        # Adjust epoch number
        if epoch % 25 == 0:
                    lr = lr*lr_decay
                    if k > 0.3:
                        k = k*k_decay
                    if lr <= 0.00000001:
                        lr = 0.00000001
                    print('Learning Rate: {:.6f}'.format(lr))
                    print('K: {:.4f}'.format(k))
                    print('Max mIoU: {:.4f}'.format(max_iou))
                    optimizer_seg, optimizer_crit = update_optimizer(lr, Segmentor.parameters(), Critic.parameters())

    plt.title("Training and Validation Losses for Critic")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(losses_C)
    plt.show()

    plt.title("Training and Validation Losses for Segmentor")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(losses_S)
    plt.show()

    plt.title("Training and Validation Accuracy for Segmentor")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(train_accuracies, val_accuracies)
    plt.show()
