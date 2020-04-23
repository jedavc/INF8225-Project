import os
import numpy as np
from glob import glob
from PIL import Image
import torch
from torchvision.transforms import *
import random

class LabelToTensor(object):

    def __init__(self):
        self.before = 255
        self.after = 1

    def __call__(self, image):
        tensor = torch.from_numpy(np.array(image)).long().unsqueeze(0)
        tensor[tensor == self.before] = self.after
        return tensor

class Dataset_train(torch.utils.data.Dataset):

    def __init__(self, path):
        self.size = (180, 135)
        self.path = path

        self.img_resize = Compose([
            Resize(self.size, Image.BILINEAR),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])
        self.label_resize = Compose([
            Resize(self.size, Image.NEAREST),
        ])
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.label_transform = Compose([
            LabelToTensor()
        ])

        #sort file names
        self.input_paths = sorted(glob(os.path.join(self.path, '{}/*.jpg'.format("ISIC-2017_Training_Data"))))
        self.label_paths = sorted(glob(os.path.join(self.path, '{}/*.png'.format("ISIC-2017_Training_Part1_GroundTruth"))))
        self.name = os.path.basename(path)

    def __getitem__(self, index):
        image = Image.open(self.input_paths[index]).convert('RGB')
        label = Image.open(self.label_paths[index]).convert('P')

        image = self.img_resize(image)
        label = self.label_resize(label)

        #flip images randomly
        image, label = self.rand_flip_image(image, label)
        #crop image to 128 x 128 dims
        image, label = self.rand_crop_image(image, label)

        image = self.img_transform(image)
        label = self.label_transform(label)
        return image, label

    #flip images
    def rand_flip_image(self, image, label):

        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)

        return image, label

    # crop image to 128 x 128 dims
    def rand_crop_image(self, image, label):

        w, h = image.size
        th, tw = (128, 128)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        if not (w == tw and h == th):
            if random.random() > 0.5:
                image = image.resize((128, 128), Image.BILINEAR)
                label = label.resize((128, 128), Image.NEAREST)
            else:
                image = image.crop((x1, y1, x1 + tw, y1 + th))
                label = label.crop((x1, y1, x1 + tw, y1 + th))

        return image, label

    def __len__(self):
        return len(self.input_paths)


class Dataset_val_test(torch.utils.data.Dataset):
    def __init__(self, data_dir, isTest):

        self.size = (128, 128)
        self.data_dir = data_dir

        self.img_transform = Compose([
            Resize(self.size, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.label_transform = Compose([
            Resize(self.size, Image.NEAREST),
            LabelToTensor()
        ])

        if isTest:
            self.input_path = sorted(glob(os.path.join(self.data_dir, '{}/*.jpg'.format("ISIC-2017_Test_v2_Data"))))
            self.label_path = sorted(glob(os.path.join(self.data_dir, '{}/*.png'.format("ISIC-2017_Test_v2_Part1_GroundTruth"))))
        if not isTest:
            self.input_path = sorted(glob(os.path.join(self.data_dir, '{}/*.jpg'.format("ISIC-2017_Validation_Data"))))
            self.label_path = sorted(glob(os.path.join(self.data_dir, '{}/*.png'.format("ISIC-2017_Validation_Part1_GroundTruth"))))

        self.name = os.path.basename(data_dir)

    def __getitem__(self, index):
        image = Image.open(self.input_path[index]).convert('RGB')
        label = Image.open(self.label_path[index]).convert('P')
        image = self.img_transform(image)
        label = self.label_transform(label)
        return image, label

    def __len__(self):
        return len(self.input_path)


def loader(dataset, batch_size, num_workers=4, shuffle=True):
    input_images = dataset
    input_loader = torch.utils.data.DataLoader(dataset=input_images,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                drop_last=True)
    return input_loader
