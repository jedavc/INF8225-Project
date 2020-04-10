from architectures.DANet.model.MSDualGuided import *
from torch.utils.data import DataLoader
from datasets.ChaosDataset import *
from torchvision import transforms
from torch.optim import Adam
from architectures.DANet.utils.MSDualGuidedLoss import *
from tqdm import tqdm
from architectures.DANet.utils.metrics import *
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    chaos_dataset = ChaosDataset(mode="train", root_dir="../rawdata/CHAOS_", transform=transform, augment=Augment())
    train_loader = DataLoader(chaos_dataset, batch_size=2, num_workers=0, shuffle=True)

    net = MSDualGuided()
    loss_module = MSDualGuidedLoss()
    Dice_loss = computeDiceOneHot()
    softMax = nn.Softmax()

    if torch.cuda.is_available():
        net.cuda()
        loss_module.cuda()
        Dice_loss.cuda()
        softMax.cuda()

    optimizer = Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99), amsgrad=False)

    for i in range(10):
        with tqdm(total=len(train_loader)) as epoch_pbar:
            epoch_pbar.set_description(f'Epoch {i}')

            net.train()
            loss_train = []
            for (image, mask) in train_loader:
                image, mask = image.cuda(), mask.cuda()

                semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2 = net(image)

                # ICI A MODIF
                segmentation_prediction = sum(list(predict1) + list(predict2)) / 8
                predClass_y = softMax(segmentation_prediction)
                segmentation_prediction_ones = predToSegmentation(predClass_y)

                Segmentation_planes = getOneHotSegmentation(mask)
                Segmentation_class = getTargetSegmentation(mask)
                DicesN, DicesB, DicesW, DicesT, DicesZ = Dice_loss(segmentation_prediction_ones, Segmentation_planes)

                DiceB = DicesToDice(DicesB)
                DiceW = DicesToDice(DicesW)
                DiceT = DicesToDice(DicesT)
                DiceZ = DicesToDice(DicesZ)
                Dice_score = (DiceB + DiceW + DiceT + DiceZ) / 4
                #

                optimizer.zero_grad()

                loss = loss_module(semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2, mask)
                loss.backward()

                optimizer.step()

                loss_train.append(loss.cpu().data.numpy())

                epoch_pbar.set_postfix_str(" Mean Dice: {:.4f}, Dice1: {:.4f}, Dice2: {:.4f}, Dice3: {:.4f}, Dice4: {:.4f} ".format(
                    Dice_score.cpu().data.numpy(),
                    DiceB.data.cpu().data.numpy(),
                    DiceW.data.cpu().data.numpy(),
                    DiceT.data.cpu().data.numpy(),
                    DiceZ.data.cpu().data.numpy()))
                epoch_pbar.update()




        print("Epoch {}: loss -> {}".format(i+1, np.mean(loss_train)))

