from architectures.NVDLMED.model.NVDLMED import *
from datasets.BrainTumorDataset import *
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import *
from architectures.NVDLMED.utils.NVDLMEDLoss import *
from tqdm import tqdm
from architectures.NVDLMED.utils.metrics import *

if __name__ == "__main__":
    desired_resolution = (80, 96, 64)
    factor = (desired_resolution[0] / 155, desired_resolution[1] / 240, desired_resolution[2] / 240)

    brain_tumor_dataset = BrainTumorDataset(
        training_path="../rawdata/MICCAI_BraTS_2018_Data_Training/",
        desired_resolution=desired_resolution,
        number_modality=4,
        original_resolution=(155, 240, 240),
        output_channels=3,
        transform_input=transforms.Compose([Resize(factor), Normalize()]),
        transform_gt=transforms.Compose([Labelize(), Resize(factor, mode="nearest")]))

    training_exemples = int(0.75 * len(brain_tumor_dataset))
    train_dataset, valid_dataset = random_split(
        brain_tumor_dataset,
        [training_exemples, len(brain_tumor_dataset) - training_exemples])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=8)

    model = NVDLMED(input_shape=(4,) + desired_resolution)
    model.cuda()

    NVDLMED_loss = NVDLMEDLoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    epochs = 150
    lambda1 = lambda epoch: (1 - epoch / epochs) ** 0.9
    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

    train_loss = []
    for epoch in range(epochs):

        # # Training loop
        # model.train()
        # with tqdm(total=len(train_loader), ascii=True) as training_bar:
        #     training_bar.set_description(f'[Training] Epoch {epoch + 1}')
        #
        #     for (input, target) in train_loader:
        #         input, target = input.cuda(), target.cuda()
        #
        #         optimizer.zero_grad()
        #         output_gt, output_vae, mu, logvar = model(input)
        #         loss, l_dice, l_l2, l_kl = NVDLMED_loss(output_gt, target, output_vae, input, mu, logvar)
        #
        #         loss.backward()
        #         optimizer.step()
        #
        #         train_loss.append(loss.item())
        #         training_bar.set_postfix_str(
        #             "Loss: {:.3f} | Dice loss: {:.3f}, L2 loss: {:.3f}, KL loss {:.3f}"
        #                 .format(loss.item(), l_dice.item(), l_l2.item(), l_kl.item()))
        #         training_bar.update()

        # Validation loop
        model.eval()
        with tqdm(total=len(valid_loader), ascii=True) as val_bar:
            val_bar.set_description('[Validation]')

            for (input, target) in valid_loader:
                input, target = input.cuda(), target.cuda()

                with torch.no_grad():
                    output_gt = model(input)
                    dsc_3d, hd_3d = calculate_3d_metrics(output_gt, target)

        scheduler.step()
