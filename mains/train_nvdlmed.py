from architectures.NVDLMED.model.NVDLMED import *
from datasets.BrainTumorDataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from architectures.NVDLMED.model.metrics import *

if __name__ == "__main__":
    desired_resolution = (96, 112, 80)
    factor = (desired_resolution[0]/155, desired_resolution[1]/240, desired_resolution[2]/240)

    brain_tumor_dataset = BrainTumorDataset(
        training_path="../rawdata/MICCAI_BraTS_2018_Data_Training/",
        desired_resolution=desired_resolution,
        number_modality=4,
        original_resolution=(155, 240, 240),
        output_channels=3,
        transform_input=transforms.Compose([Resize(factor), Normalize()]),
        transform_gt=transforms.Compose([Labelize(), Resize(factor, mode="nearest")]))

    model = NVDLMED(input_shape=(4,)+desired_resolution)
    model.cuda()
    optimizer = Adam(model.parameters(), lr=1e-4)

    train_loader = DataLoader(brain_tumor_dataset, batch_size=1, shuffle=False, num_workers=2)
    model.train()
    for epoch in range(10):
        print("Epoch #{}".format(epoch))
        for (input, target) in train_loader:
            input, target = input.cuda(), target.cuda()
            output_gt, output_vae = model(input)
            optimizer.zero_grad()
            loss = loss_gt(target, output_gt)
            print(loss.item())
            loss.backward()
            optimizer.step()
