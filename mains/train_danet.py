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
    val_loader = DataLoader(val_chaos_dataset, batch_size=1, num_workers=0, shuffle=True)

    net = MSDualGuided().cuda()
    loss_module = MSDualGuidedLoss()
    optimizer = Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99), amsgrad=False)

    for i in range(150):

        # Training Loop
        with tqdm(total=len(train_loader), ascii=True) as training_bar:
            training_bar.set_description(f'[Training] Epoch {i}')

            net.train()
            loss_train = []
            for (image, mask, _) in train_loader:
                image, mask = image.cuda(), mask.cuda()
                semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2 = net(image)

                optimizer.zero_grad()
                loss = loss_module(semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2, mask)

                loss.backward()
                optimizer.step()

                loss_train.append(loss.cpu().data.numpy())

                segmentation_prediction = sum(list(predict1) + list(predict2)) / 8
                classes_dice = dice_score(segmentation_prediction, mask)

                training_bar.set_postfix_str(
                    "Mean dice: {:.3f} || Liver: {:.3f}, Kidney(R): {:.3f}, Kidney(L): {:.3f}, Spleen: {:.3f}"
                    .format(torch.mean(classes_dice[1:]), classes_dice[1], classes_dice[2], classes_dice[3], classes_dice[4])
                )
                training_bar.update()

        # print("\nEpoch {}: loss -> {}\n".format(i + 1, np.mean(loss_train)))

        # Validation Loop
        with tqdm(total=len(val_loader), ascii=True) as val_bar:
            val_bar.set_description('[Validation]')

            net.eval()
            dice_val = torch.zeros(len(val_loader), 5)
            for i, (val_image, val_mask, val_img_name) in enumerate(val_loader):
                val_image, val_mask = val_image.cuda(), val_mask.cuda()

                with torch.no_grad():
                    seg_pred = net(val_image)
                    dice_val[i] = dice_score(seg_pred, val_mask)

                    prediction_to_png(seg_pred, val_img_name)

                val_bar.update()

            create_3d_volume("../rawdata/CHAOS_/val/Result", "../rawdata/CHAOS_/val/Volume/Pred")
            dice_3d = dice_score_3d("../rawdata/CHAOS_/val/Volume")

        print(dice_val.mean(dim=0))
