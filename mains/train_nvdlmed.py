from architectures.NVDLMED.model.NVDLMED import *
from datasets.BrainTumorDataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from architectures.NVDLMED.model.metrics import *

if __name__ == "__main__":
    brain_tumor_dataset = BrainTumorDataset(
        training_path="../rawdata/MICCAI_BraTS_2018_Data_Training/",
        desired_resolution=(80, 96, 64),
        original_resolution=(155, 240, 240),
        output_channels=3,
        transform_input=transforms.Compose([Resize((80/155, 96/240, 64/240)), Normalize()]),
        transform_gt=transforms.Compose([Labelize(), Resize((80/155, 96/240, 64/240), mode="nearest")]))

    model = NVDLMED(input_shape=(4, 80, 96, 64))
    model.cuda()
    optimizer = Adam(model.parameters(), lr=1e-4)

    train_loader = DataLoader(brain_tumor_dataset, batch_size=1, shuffle=False)
    model.train()
    for (input, target) in train_loader:
        input, target = input.cuda(), target.cuda()
        output_gt, output_vae = model(input)
        optimizer.zero_grad()
        loss = loss_gt(target, output_gt)
        print(loss.item())
        loss.backward()
        optimizer.step()
