from architectures.NVDLMED.model.NVDLMED import *
from datasets.BrainTumorDataset import *
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import *
from architectures.NVDLMED.utils.NVDLMEDLoss import *
from tqdm import tqdm
from architectures.NVDLMED.utils.metrics import *
import warnings
import argparse
import shutil
from architectures.NVDLMED.utils.HierarchyCreator import create_hierarchy

warnings.filterwarnings("ignore")


def run_training(args):
    desired_resolution = (80, 96, 64)
    factor = (desired_resolution[0] / 155, desired_resolution[1] / 240, desired_resolution[2] / 240)

    train_dataset = BrainTumorDataset(
        mode="train",
        data_path="../rawdata/brats/",
        desired_resolution=desired_resolution,
        original_resolution=(155, 240, 240),
        transform_input=transforms.Compose([Resize(factor), Normalize()]),
        transform_gt=transforms.Compose([Labelize(), Resize(factor, mode="nearest")]))

    val_dataset = BrainTumorDataset(
        mode="val",
        data_path="../rawdata/brats/",
        desired_resolution=desired_resolution,
        original_resolution=(155, 240, 240),
        transform_input=transforms.Compose([Resize(factor), Normalize()]),
        transform_gt=transforms.Compose([Labelize(), Resize(factor, mode="nearest")]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = NVDLMED(input_shape=(4,) + desired_resolution)
    model.cuda()

    NVDLMED_loss = NVDLMEDLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    lambda1 = lambda epoch: (1 - epoch / args.epochs) ** 0.9
    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

    lossG = []
    dsc = []
    hd = []

    best_dice_3d, best_epoch = 0, 0
    for epoch in range(args.epochs):

        # Training loop
        model.train()
        loss_train = 0
        with tqdm(total=len(train_loader), ascii=True) as training_bar:
            training_bar.set_description(f'[Training] Epoch {epoch + 1}')

            for (input, target) in train_loader:
                input, target = input.cuda(), target.cuda()

                optimizer.zero_grad()
                output_gt, output_vae, mu, logvar = model(input)
                loss, l_dice, l_l2, l_kl = NVDLMED_loss(output_gt, target, output_vae, input, mu, logvar)

                loss.backward()
                optimizer.step()

                loss_train += loss.item()
                training_bar.set_postfix_str(
                    "Loss: {:.3f} | Dice loss: {:.3f}, L2 loss: {:.3f}, KL loss {:.3f}"
                        .format(loss.item(), l_dice.item(), l_l2.item(), l_kl.item()))
                training_bar.update()

            training_bar.set_postfix_str("Mean loss: {:.4f}".format(loss_train / len(train_loader)))

        # Validation loop
        model.eval()
        dsc_valid, hd_valid = [], []
        with tqdm(total=len(valid_loader), ascii=True) as val_bar:
            val_bar.set_description('[Validation]')

            for (input, target) in valid_loader:
                input, target = input.cuda(), target.cuda()

                with torch.no_grad():
                    output_gt = model(input)

                    dsc_3d, hd_3d = calculate_3d_metrics(output_gt, target)
                    dsc_valid.extend(dsc_3d)
                    hd_valid.extend(hd_3d)

                val_bar.update()

            val_bar.set_postfix_str(
                "Dice 3D: {:.3f} || ET: {:.3f}, WT: {:.3f}, TC: {:.3f}"
                    .format(np.mean(dsc_valid), np.mean(dsc_valid, 0)[0], np.mean(dsc_valid, 0)[1], np.mean(dsc_valid, 0)[2])
            )

        current_dice_3d = np.mean(dsc_valid)
        if current_dice_3d > best_dice_3d:
            best_dice_3d = current_dice_3d
            torch.save(model.state_dict(), args.root_dir + "/save/net.pth")

        scheduler.step()

        # Save Statistics
        lossG.append(loss_train / len(train_loader))
        dsc.append(dsc_valid)
        hd.append(hd_valid)

        np.save(args.root_dir + "/save/loss", lossG)
        np.save(args.root_dir + "/save/dsc", dsc)
        np.save(args.root_dir + "/save/assd", hd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='../rawdata/MICCAI_BraTS_2018_Data_Training', type=str)
    parser.add_argument('--root_dir', default='../rawdata/brats', type=str)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--create_hierarchy', default=False, action='store_true')
    parser.add_argument('--checkpoint_path', default='../rawdata/brats/save/net.pth', type=str)

    args = parser.parse_args()

    if args.create_hierarchy:
        print("Creating folders for the model!\n")
        shutil.rmtree(args.root_dir, ignore_errors=True)
        create_hierarchy(data_dir=args.data_dir, out_dir=args.root_dir)

    if args.train:
        run_training(args)

    # if args.eval:
    #     run_eval(args)