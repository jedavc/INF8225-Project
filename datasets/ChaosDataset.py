from torch.utils.data import Dataset
from PIL import Image, ImageOps
from random import random
import os
import glob
import torch


class ChaosDataset(Dataset):
    def __init__(self,
                 mode,
                 root_dir,
                 transform_input=None,
                 transform_mask=None,
                 augment=None,
                 equalize=False):

        self.root_dir = root_dir
        self.files = self.load_files(root_dir, mode)

        self.transform_input = transform_input
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

        if self.transform_input:
            img = self.transform_input(img)
            mask_t = self.transform_input(mask)

        if self.transform_mask:
            mask = self.transform_mask(mask)

        return img, mask_t, img_path, torch.from_numpy(mask)


class GrayToClass(object):

    def __init__(self):
        self.class1 = 63
        self.class2 = 126
        self.class3 = 189
        self.class4 = 252

    def __call__(self, mask):
        numpy_image = np.array(mask)

        numpy_image = np.where(numpy_image == self.class1, 1, numpy_image)
        numpy_image = np.where(numpy_image == self.class2, 2, numpy_image)
        numpy_image = np.where(numpy_image == self.class3, 3, numpy_image)
        numpy_image = np.where(numpy_image == self.class4, 4, numpy_image)

        return numpy_image


class Augment(object):

    def __init__(self, prob=0.5):
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

import pydicom
import cv2
import numpy as np
import shutil

def create_image_dataset(root_dir="../rawdata/CHAOS_Train_Sets/Train_Sets/MR", out_dir="../rawdata/CHAOS_"):
    try:
        os.makedirs(out_dir + "/train")
        os.makedirs(out_dir + "/train/Img")
        os.makedirs(out_dir + "/train/GT")
        os.makedirs(out_dir + "/val")
        os.makedirs(out_dir + "/val/Img")
        os.makedirs(out_dir + "/val/GT")
        os.makedirs(out_dir + "/test")
        os.makedirs(out_dir + "/test/Img")
        os.makedirs(out_dir + "/test/GT")
    except:
        pass

    nb_patient = os.listdir(root_dir)
    for i, no_patient in enumerate(nb_patient):
        dcm_path = os.path.join(root_dir, no_patient, "T1DUAL/DICOM_anon/InPhase")
        gt_path = os.path.join(root_dir, no_patient, "T1DUAL/Ground")

        dcm_files = glob.glob(dcm_path + "/*.dcm")
        gt_files = glob.glob(gt_path + "/*.png")

        for j, dcm_file in enumerate(dcm_files):
            ds = pydicom.dcmread(dcm_file)
            pixel_array_numpy = ds.pixel_array.astype(float)
            img = Image.open(gt_files[j])

            #Center crop dcm
            if pixel_array_numpy.shape[0] > 256:
                centerX = int(pixel_array_numpy.shape[0] / 2)
                centerY = int(pixel_array_numpy.shape[1] / 2)
                newImage = pixel_array_numpy[centerX - 128:centerX + 128, centerY - 128:centerY + 128]
                pixel_array_numpy = newImage

            # Center crop png
            if img.size[0] > 256:
                w, h = img.size
                left = (w - 256) / 2
                top = (h - 256) / 2
                right = (w + 256) / 2
                bottom = (h + 256) / 2
                img = img.crop((left, top, right, bottom))

            #Normalise to 0-255
            pixel_array_numpy_gray = (pixel_array_numpy - np.min(pixel_array_numpy)) / (np.max(pixel_array_numpy) - np.min(pixel_array_numpy))*255

            name = "Subj_" + no_patient + "slice_" + str(j + 1) + ".png"
            cv2.imwrite(os.path.join(out_dir, "train/Img", name), pixel_array_numpy_gray.astype('uint8'))
            img.save(os.path.join(out_dir, "train/GT", name))
