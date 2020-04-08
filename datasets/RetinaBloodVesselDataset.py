import numpy as np
import os
from PIL import Image

class RetinaBloodVesselDataset():

    def __init__(self,
                 training_images_path="../rawdata/DRIVE/training/images/",
                 training_groundTruth_path="../rawdata/DRIVE/training/1st_manual/",
                 training_borderMasks_path = "../rawdata/DRIVE/training/mask/",
                 testing_images_path="../rawdata/DRIVE/test/images/",
                 testing_groundTruth_path="../rawdata/DRIVE/test/1st_manual/",
                 testing_borderMasks_path="../rawdata/DRIVE/test/mask/",
                 n_images=20,
                 n_channels=3,
                 height=584,
                 width=565
                 ):
        self.training_images_path = training_images_path
        self.training_groundTruth_path = training_groundTruth_path
        self.training_borderMasks_path = training_borderMasks_path
        self.testing_images_path = testing_images_path
        self.testing_groundTruth_path = testing_groundTruth_path
        self.testing_borderMasks_path = testing_borderMasks_path
        self.n_images = n_images
        self.n_channels = n_channels
        self.height = height
        self.width = width
        self.inputs_train, self.gt_train, self.bm_train = self.extract_datasets(self.training_images_path,
                                                                                self.training_groundTruth_path,
                                                                                self.training_borderMasks_path,
                                                                                "train")
        self.inputs_test, self.gt_test, self.bm_test = self.extract_datasets(self.testing_images_path,
                                                                                self.testing_groundTruth_path,
                                                                                self.testing_borderMasks_path,
                                                                                "test")


    def extract_datasets(self, inputs_path , ground_truth_path, border_masks_path, dataset_type = ""):
        inputs = np.empty((self.n_images, self.height, self.width, self.n_channels))
        groundTruth_values = np.empty((self.n_images, self.height, self.width))
        border_masks_values = np.empty((self.n_images, self.height, self.width))
        for path, sub_directory, files in os.walk(inputs_path):  # list all files, directories in the path
            for i in range(len(files)):
                # original
                print("original image: " + files[i])
                image = Image.open(inputs_path + files[i])
                inputs[i] = np.asarray(image)
                # corresponding ground truth
                image_gT_file = files[i][0:2] + "_manual1.gif"
                print("ground truth name: " + image_gT_file)
                image_ground_truth = Image.open(ground_truth_path + image_gT_file)
                groundTruth_values[i] = np.asarray(image_ground_truth)
                # corresponding border masks
                image_bM_file = ""
                if dataset_type == "train":
                    image_bM_file = files[i][0:2] + "_training_mask.gif"
                elif dataset_type == "test":
                    image_bM_file = files[i][0:2] + "_test_mask.gif"
                else:
                    print("Train or test must be the value of the dataset_type")
                    exit()
                print("border masks name: " + image_bM_file)
                image_border_mask = Image.open(border_masks_path + image_bM_file)
                border_masks_values[i] = np.asarray(image_border_mask)
        return self.reshape_inputs(inputs), \
               self.reshape_ground_truth(groundTruth_values), \
               self.reshape_border_masks(border_masks_values)

    def reshape_inputs(self, inputs):
        return np.transpose(inputs, (0,3,1,2))

    def reshape_ground_truth(self, ground_truth):
        return np.reshape(ground_truth,(self.n_images, 1, self.height, self.width))

    def reshape_border_masks(self, border_masks):
        return np.reshape(border_masks, (self.n_images, 1, self.height, self.width))

    def get_training_data(self):
        return self.extract_datasets(self.training_images_path,
                                     self.training_groundTruth_path,
                                     self.training_borderMasks_path, "train")


