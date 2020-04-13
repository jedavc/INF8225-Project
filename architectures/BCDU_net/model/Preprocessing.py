import numpy as np
import cv2
import random
from datasets.RetinaBloodVesselDataset import *
from architectures.BCDU_net.model.PatchesExtraction import *
import math


class Preprocessing():
    def __init__(self):
        self.new_height = None
        self.new_width = None
        pass

    def run_preprocess_pipeline(self, data_input, type, data_bm=None):
        gray_input = self.RGB2gray(data_input)
        normalized_input = self.normalize(gray_input)
        tiles_input = self.divide_to_tiles(normalized_input)
        good_gamma_input = self.fix_gamma(tiles_input, 1.2)
        reduced_range_input = self.reduce_range(good_gamma_input)

        if data_bm is not None:
            patches_extractor = PatchesExtraction()
            reduced_range_masks = self.reduce_range(data_bm)
            if type == "train":
                no_bottom_top_input = self.cut_bottom_top(reduced_range_input)
                no_bottom_top_bm = self.cut_bottom_top(reduced_range_masks)
                input_patches, bm_patches = patches_extractor.rand_extract_patches(no_bottom_top_input,
                                                                      no_bottom_top_bm,
                                                                      64,
                                                                      64,
                                                                      200000)
            else:
                extended_input = self.extend_images(reduced_range_input)
                bm_patches = self.extend_images(reduced_range_masks)
                removed_overlap_input = patches_extractor.remove_overlap(extended_input)
                self.new_height = removed_overlap_input.shape[2]
                self.new_width = removed_overlap_input.shape[3]
                input_patches = patches_extractor.view_patches(removed_overlap_input)
        else:
            input_patches = reduced_range_input
            bm_patches = ""
        return input_patches, bm_patches

    def new_dimensions(self):
        return self.new_height, self.new_width

    def RGB2gray(self, data):
        channel_1 = data[:, 0, :, :]
        channel_2 = data[:, 1, :, :]
        channel_3 = data[:, 2, :, :]
        channel_1 = channel_1 * 0.299
        channel_2 = channel_2 * 0.587
        channel_3 = channel_3 * 0.114
        black_white_images = channel_1 + channel_2 + channel_3
        black_white_images = np.reshape(black_white_images,
                                        (data.shape[0],
                                         1,
                                         data.shape[2],
                                         data.shape[3]))
        return black_white_images
#--------------------------------------------------------------------------------------------------------------------------------------------------------
    def rgb2gray(self, rgb):
        assert (len(rgb.shape) == 4)  # 4D arrays
        assert (rgb.shape[1] == 3)
        bn_imgs = rgb[:, 0, :, :] * 0.299 + rgb[:, 1, :, :] * 0.587 + rgb[:, 2, :, :] * 0.114
        bn_imgs = np.reshape(bn_imgs, (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))
        return bn_imgs

    def dataset_normalized(self, imgs):
        assert (len(imgs.shape) == 4)  # 4D arrays
        assert (imgs.shape[1] == 1)  # check the channel is 1
        imgs_normalized = np.empty(imgs.shape)
        imgs_std = np.std(imgs)
        imgs_mean = np.mean(imgs)
        imgs_normalized = (imgs - imgs_mean) / imgs_std
        for i in range(imgs.shape[0]):
            imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                        np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
        return imgs_normalized

    def adjust_gamma(self, imgs, gamma=1.0):
        assert (len(imgs.shape) == 4)  # 4D arrays
        assert (imgs.shape[1] == 1)  # check the channel is 1
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        new_imgs = np.empty(imgs.shape)
        for i in range(imgs.shape[0]):
            new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype=np.uint8), table)
        return new_imgs

    def is_patch_inside_FOV(self, x, y, img_w, img_h, patch_h):
        x_ = x - int(img_w / 2)  # origin (0,0) shifted to image center
        y_ = y - int(img_h / 2)  # origin (0,0) shifted to image center
        R_inside = 270 - int(patch_h * np.sqrt(
            2.0) / 2.0)  # radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
        radius = np.sqrt((x_ * x_) + (y_ * y_))
        if radius < R_inside:
            return True
        else:
            return False

    def extract_random(self, full_imgs, full_masks, patch_h, patch_w, N_patches, inside=True):
        if (N_patches % full_imgs.shape[0] != 0):
            print("N_patches: plase enter a multiple of 20")
            exit()
        assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 4)  # 4D arrays
        assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
        assert (full_masks.shape[1] == 1)  # masks only black and white
        assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
        patches = np.empty((N_patches, full_imgs.shape[1], patch_h, patch_w))
        patches_masks = np.empty((N_patches, full_masks.shape[1], patch_h, patch_w))
        img_h = full_imgs.shape[2]  # height of the full image
        img_w = full_imgs.shape[3]  # width of the full image
        # (0,0) in the center of the image
        patch_per_img = int(N_patches / full_imgs.shape[0])  # N_patches equally divided in the full images
        print("patches per full image: " + str(patch_per_img))
        iter_tot = 0  # iter over the total numbe rof patches (N_patches)
        for i in range(full_imgs.shape[0]):  # loop over the full images
            k = 0
            while k < patch_per_img:
                x_center = random.randint(0 + int(patch_w / 2), img_w - int(patch_w / 2))
                # print "x_center " +str(x_center)
                y_center = random.randint(0 + int(patch_h / 2), img_h - int(patch_h / 2))
                # print "y_center " +str(y_center)
                # check whether the patch is fully contained in the FOV
                if inside == True:
                    if self.is_patch_inside_FOV(x_center, y_center, img_w, img_h, patch_h) == False:
                        continue
                patch = full_imgs[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                        x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
                patch_mask = full_masks[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                             x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
                patches[iter_tot] = patch
                patches_masks[iter_tot] = patch_mask
                iter_tot += 1  # total
                k += 1  # per full_img
        return patches, patches_masks

    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    def normalize(self, data):
        normalized_data = np.empty(data.shape)
        std = np.std(data)
        mean = np.mean(data)
        normalized_data = (data - mean) / std
        for image in range(data.shape[0]):
            min = np.min(normalized_data[image])
            max = np.max(normalized_data[image])
            normalized_data[image] = ((normalized_data[image] - min) / (max - min)) * 255
        return normalized_data

    def divide_to_tiles(self, data):
        CLAHE_object = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        tiles_data = np.empty(data.shape)
        for image in range(data.shape[0]):
            applied_CLAHE = CLAHE_object.apply(np.array(data[image, 0], dtype=np.uint8))
            tiles_data[image, 0] = applied_CLAHE
        return tiles_data

    def fix_gamma(self, data, gamma=1.0):
        adjusted_gammas = self.map_pixel_gamma(gamma)
        corrected_data = np.empty(data.shape)
        for image in range(data.shape[0]):
            corrected_data[image, 0] = cv2.LUT(np.array(data[image, 0], dtype=np.uint8), adjusted_gammas)
        return corrected_data

    def map_pixel_gamma(self, gamma):
        inverted_gamma = 1.0 / gamma
        fixed_gammas = []
        for pixel in range(0, 256):
            fixed_gamma = ((pixel / 255.0) ** inverted_gamma) * 255
            fixed_gammas.append(fixed_gamma)
        return np.array(fixed_gammas).astype("uint8")

    def reduce_range(self, data):
        return data / 255

    def cut_bottom_top(self, data):
        return data[:, :, 9:574, :]

    # def is_fully_contained(self, x, y, width, height, patch_height):
    #     x_ = x - int(width / 2)  # origin (0,0) shifted to image center
    #     y_ = y - int(height / 2)  # origin (0,0) shifted to image center
    #     in_radius = 270 - int(patch_height * np.sqrt(
    #         2.0) / 2.0)  # radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
    #     radius = np.sqrt((x_ * x_) + (y_ * y_))
    #     if radius < in_radius:
    #         return True
    #     else:
    #         return False

    def extend_images(self, data, n_tests=20):
        return data[0:n_tests, :, :, :]


