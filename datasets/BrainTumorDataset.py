import SimpleITK as sitk
import torch
import re
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import glob
import random
import numpy as np


class BrainTumorDataset(Dataset):

    def __init__(self,
                 training_path="../rawdata/MICCAI_BraTS_2018_Data_Training/",
                 desired_resolution=(80, 96, 64),
                 original_resolution=(155, 240, 240),
                 output_channels=3,
                 transform_input=None,
                 transform_gt=None):
        self.training_path = training_path
        self.training_files = {"t1": glob.glob(training_path + '*GG/*/*t1.nii.gz'),
                               "t2": glob.glob(training_path + '*GG/*/*t2.nii.gz'),
                               "flair": glob.glob(training_path + '*GG/*/*flair.nii.gz'),
                               "t1ce": glob.glob(training_path + '*GG/*/*t1ce.nii.gz'),
                               "seg": glob.glob(training_path + '*GG/*/*seg.nii.gz')}

        self.desired_resolution = desired_resolution
        self.original_resolution = original_resolution
        self.output_channels = output_channels
        self.transform_input = transform_input
        self.transform_gt = transform_gt
        self.files = self.find_files()

    def find_files(self):
        path = re.compile('.*_(\w*)\.nii\.gz')
        data_paths = [{
            path.findall(item)[0]: item
            for item in items
        }
            for items in list(zip(self.training_files["t1"], self.training_files["t2"], self.training_files["t1ce"], self.training_files["flair"], self.training_files["seg"]))]

        return data_paths

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        data_files = self.files[idx]
        numpy_data = np.array([sitk.GetArrayFromImage(sitk.ReadImage(file))
                               for file in data_files.values()], dtype=np.float32)

        input = numpy_data[0:4]
        if self.transform_input is not None:
            input = self.transform_input(numpy_data[0:4])

        gt = self.transform_gt(numpy_data[-1])

        return torch.from_numpy(input), torch.from_numpy(gt)


class Resize(object):
    def __init__(self, factors, mode='constant', dtype=np.float32):
        self.factors = factors
        self.mode = mode
        self.dtype = dtype

    def __call__(self, data):
        assert len(self.factors) == 3
        resized_data = np.array([zoom(data[i], self.factors, mode=self.mode)
                                 for i in range(data.shape[0])], dtype=self.dtype)

        return resized_data


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, data):
        normalized_data = np.array([(data[i] - data[i].mean()) / data[i].std()
                                    for i in range(data.shape[0])], dtype=np.float32)

        return normalized_data


class Labelize(object):
    def __init__(self):
        pass

    def __call__(self, data):
        ncr = data == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
        ed = data == 2  # Peritumoral Edema (ED)
        et = data == 4  # GD-enhancing Tumor (ET)

        return np.array([ncr, ed, et], dtype=np.uint8)

class CropCenter3DInput(object):
    def __init__(self, cropx, cropy, cropz):
        self.cropx = cropx
        self.cropy = cropy
        self.cropz = cropz
        self.startx = 0
        self.starty = 0
        self.startz = 0

    def set_new_dimensions(self, data):
        x, y, z = data.shape
        self.startx = x // 2 - (self.cropx // 2)
        self.starty = y // 2 - (self.cropy // 2)
        self.startz = z // 2 - (self.cropz // 2)
        return data[self.startx:self.startx + self.cropx, self.starty:self.starty + self.cropy, self.startz:self.startz + self.cropz]

    def __call__(self, data):
        cropped_data = np.array([self.set_new_dimensions(data[i])
                                 for i in range(1, data.shape[0])], dtype=np.float32)
        print("Center cropped:")
        print(cropped_data[1].shape)
        return cropped_data

class RandomlyCropInput(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def set_new_dimensions(self, data):
        print(data.shape)
        x, y, z = data.shape
        self.starty = random.randint(0, y - self.width)
        self.startz = random.randint(0, z - self.height)
        for slice in data:
            print(slice.shape)
            slice = slice[self.starty:self.starty + self.width,
               self.startz:self.startz + self.height]

        return data

    def __call__(self, data):
        cropped_data = np.array([self.set_new_dimensions(data[i])
                                 for i in range(data.shape[0])], dtype=np.float32)
        print("Randomly cropped:")
        print(cropped_data[1].shape)
        return cropped_data

class CropCenter3DOutput(object):
    def __init__(self, cropx, cropy, cropz):
        self.cropx = cropx
        self.cropy = cropy
        self.cropz = cropz
        self.startx = 0
        self.starty = 0
        self.startz = 0

    def set_new_dimensions(self, data):
        x, y, z = data.shape
        self.startx = x // 2 - (self.cropx // 2)
        self.starty = y // 2 - (self.cropy // 2)
        self.startz = z // 2 - (self.cropz // 2)
        return data[self.startx:self.startx + self.cropx, self.starty:self.starty + self.cropy, self.startz:self.startz + self.cropz]

    def __call__(self, data):
        cropped_data = np.array(self.set_new_dimensions(data), dtype=np.float32)
        print("Center cropped:")
        print(cropped_data[1].shape)
        return cropped_data

class RandomlyCropOutput(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def set_new_dimensions(self, data):
        print(data.shape)
        x, y, z = data.shape
        self.starty = random.randint(0, y - self.width)
        self.startz = random.randint(0, z - self.height)
        for slice in data:
            print(slice.shape)
            slice = slice[self.starty:self.starty + self.width,
               self.startz:self.startz + self.height]

        return data

    def __call__(self, data):
        cropped_data = np.array(self.set_new_dimensions(data), dtype=np.float32)
        print("Randomly cropped:")
        print(cropped_data[1].shape)
        return cropped_data