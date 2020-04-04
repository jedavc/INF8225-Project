from architectures.SegAN.model import *
from datasets.BrainTumorDataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from architectures.NVDLMED.model.metrics import *

if __name__ == "__main__":
    brain_tumor_dataset = BrainTumorDataset(
        training_path = "../rawdata/MICCAI_BraTS_2018_Data_Training/",
        desired_resolution = (155, 240, 240),
        original_resolution = (155, 240, 240),
        output_channels = 3,
        transform_input = transforms.Compose([CropCenter3DInput(128, 180, 180), RandomlyCropInput(160, 160)]),
        transform_gt = transforms.Compose([CropCenter3DOutput(128, 180, 180), RandomlyCropOutput(160, 160)]))

    i, o = brain_tumor_dataset[0]
    print(i)
    train_loader = DataLoader(brain_tumor_dataset, batch_size=1, shuffle=False)


