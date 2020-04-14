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
    train_chaos_dataset = ChaosDataset(mode="train", root_dir="../rawdata/CHAOS_", transform_input=transform, transform_mask=GrayToClass(), augment=Augment())
    train_loader = DataLoader(train_chaos_dataset, batch_size=2, num_workers=0, shuffle=True)

    val_chaos_dataset = ChaosDataset(mode="val", root_dir="../rawdata/CHAOS_", transform_input=transform, transform_mask=GrayToClass(), augment=None)
    val_loader = DataLoader(val_chaos_dataset, batch_size=1, num_workers=0, shuffle=True)

    net = MSDualGuided()
    loss_module = MSDualGuidedLoss()
    softMax = nn.Softmax()

    if torch.cuda.is_available():
        net.cuda()
        loss_module.cuda()
        softMax.cuda()

    optimizer = Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99), amsgrad=False)

    for i in range(150):
        with tqdm(total=len(train_loader)) as training_bar:
            training_bar.set_description(f'[Training] Epoch {i}')

            net.train()
            loss_train = []
            for (image, mask, img_path) in train_loader:
                image, mask = image.cuda(), mask.cuda()

                optimizer.zero_grad()
                net.zero_grad()

                semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2 = net(image)

                optimizer.zero_grad()
                loss = loss_module(semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2, mask)

                loss.backward()
                optimizer.step()

                loss_train.append(loss.cpu().data.numpy())

                segmentation_prediction = sum(list(predict1) + list(predict2)) / 8
                dice = dice_score(segmentation_prediction, mask)

                training_bar.set_postfix_str("Mean Dice Score: {:.4f}".format(dice.item()))
                training_bar.update()

        print("\nEpoch {}: loss -> {}\n".format(i + 1, np.mean(loss_train)))

        with tqdm(total=len(val_loader)) as val_bar:
            val_bar.set_description('[Validation]')

            net.eval()
            dice_val = torch.zeros(len(val_loader), 1)
            for i, (val_image, val_mask, val_img_path) in enumerate(val_loader):
                val_image, val_mask = val_image.cuda(), val_mask.cuda()

                with torch.no_grad():
                    seg_pred = net(val_image)
                    predClass_y_val = softMax(seg_pred)

                    dice_val[i] = dice_score(seg_pred, val_mask)

                    saveImages_for3D(predClass_y_val, val_img_path)

                val_bar.update()

        print(dice_val.mean(dim=0))





