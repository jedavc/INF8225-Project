from architectures.DANet.model.MSDualGuided import *
from torch.utils.data import DataLoader
from datasets.ChaosDataset import *
from torchvision import transforms
from torch.optim import Adam
from architectures.DANet.utils.MSDualGuidedLoss import *
from tqdm import tqdm
from architectures.DANet.utils.metrics import *
import warnings
from architectures.DANet.utils.utils import *

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    train_chaos_dataset = ChaosDataset(mode="train", root_dir="../rawdata/CHAOS_", transform_input=transform,
                                       transform_mask=GrayToClass(), augment=Augment())
    train_loader = DataLoader(train_chaos_dataset, batch_size=2, num_workers=0, shuffle=True)

    val_chaos_dataset = ChaosDataset(mode="val", root_dir="../rawdata/CHAOS_", transform_input=transform,
                                     transform_mask=GrayToClass(), augment=None)
    val_loader = DataLoader(val_chaos_dataset, batch_size=16, num_workers=0, shuffle=True)

    net = MSDualGuided().cuda()
    loss_module = MSDualGuidedLoss()
    lr = 0.001
    optimizer = Adam(net.parameters(), lr=lr, betas=(0.9, 0.99), amsgrad=False)


    lossG = []
    dsc = []
    assd = []
    vs = []

    epochs = 150
    best_dice_3d, BestEpoch = 0, 0
    for i in range(epochs):

        # Training Loop
        with tqdm(total=len(train_loader), ascii=True) as training_bar:
            training_bar.set_description(f'[Training] Epoch {i+1}')

            net.train()
            loss_train = 0
            for (image, mask, _) in train_loader:
                image, mask = image.cuda(), mask.cuda()
                semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2 = net(image)

                optimizer.zero_grad()
                loss = loss_module(semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2, mask)

                loss.backward()
                optimizer.step()

                loss_train += loss.item()

                segmentation_prediction = sum(list(predict1) + list(predict2)) / 8
                classes_dice = dice_score(segmentation_prediction, mask)

                training_bar.set_postfix_str(
                    "Mean dice: {:.3f} || Liver: {:.3f}, Kidney(R): {:.3f}, Kidney(L): {:.3f}, Spleen: {:.3f}"
                        .format(torch.mean(classes_dice[1:]), classes_dice[1], classes_dice[2], classes_dice[3],
                                classes_dice[4])
                )
                training_bar.update()

            training_bar.set_postfix_str("Mean loss: {:.4f}".format(loss_train / len(train_loader)))
            del semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2

        # Validation Loop
        with tqdm(total=len(val_loader), ascii=True) as val_bar:
            val_bar.set_description('[Validation]')

            net.eval()
            for j, (val_image, val_mask, val_img_name) in enumerate(val_loader):
                val_image, val_mask = val_image.cuda(), val_mask.cuda()

                with torch.no_grad():
                    seg_pred = net(val_image)
                    prediction_to_png(seg_pred, val_img_name)

                val_bar.update()

            create_3d_volume("../rawdata/CHAOS_/val/Result", "../rawdata/CHAOS_/val/Volume/Pred")
            dsc_3d, assd_3d, vs_3d = calculate_3d_metrics("../rawdata/CHAOS_/val/Volume")

            current_dice_3d = np.mean(dsc_3d)
            if current_dice_3d > best_dice_3d:
                best_dice_3d = current_dice_3d
                BestEpoch = i
                torch.save(net.state_dict(), "../rawdata/CHAOS_/save/net.pth")

            if i % (BestEpoch + 50) == 0:
                for param_group in optimizer.param_groups:
                    lr = lr * 0.5
                    param_group['lr'] = lr

            dice_3d_class = np.mean(dsc_3d, 0)
            val_bar.set_postfix_str(
                "Dice 3D: {:.3f} || Liver: {:.3f}, Kidney(R): {:.3f}, Kidney(L): {:.3f}, Spleen: {:.3f}"
                    .format(np.mean(dice_3d_class), dice_3d_class[0], dice_3d_class[1], dice_3d_class[2], dice_3d_class[3])
            )

            # Save Statistics
            lossG.append(loss_train/len(train_loader))
            dsc.append(dsc_3d)
            assd.append(assd_3d)
            vs.append(vs_3d)

            np.save("../rawdata/CHAOS_/save/loss", lossG)
            np.save("../rawdata/CHAOS_/save/dsc", dsc)
            np.save("../rawdata/CHAOS_/save/assd", assd)
            np.save("../rawdata/CHAOS_/save/vs", vs)

