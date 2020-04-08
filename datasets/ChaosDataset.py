from torch.utils.data import Dataset
from PIL import Image, ImageOps
from random import random
import os


class ChaosDataset(Dataset):
    def __init__(self,
                 mode,
                 root_dir,
                 transform=None,
                 transform_mask=None,
                 augment=None,
                 equalize=False):

        self.root_dir = root_dir
        self.files = self.load_files(root_dir, mode)

        self.transform = transform
        self.transform_mask = transform_mask
        self.augment = augment
        self.equalize = equalize

    @staticmethod
    def load_files(root_dir, mode):
        assert mode in ["train", "val", "test"]
        files = []
        img_path = os.path.join(root_dir, mode, "Img")
        mask_path = os.path.join(root_dir, mode, "GT")

        images = os.listdir(img_path)
        images.sort()
        masks = os.listdir(mask_path)
        masks.sort()
        
        for img, mask in zip(images, masks):
            file = (os.path.join(img_path, img), os.path.join(mask_path, mask))
            files.append(file)

        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, mask_path = self.files[idx]
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            img, mask = self.augment(img, mask)

        if self.transform:
            img = self.transform(img)

        if self.transform_mask:
            mask = self.transform_mask(mask)

        return img, mask, img_path


class Augment(object):
    def __int__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask):
        if random() > self.prob:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)

        if random() > self.prob:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)

        if random() > self.prob:
            angle = random() * 60 - 30
            img = img.rotate(angle)
            mask = mask.rotate(angle)

        return img, mask