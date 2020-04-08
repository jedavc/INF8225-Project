import numpy as np
import cv2
import random

class Preprocessing():
    def __init__(self):
        pass


    def run_preprocess_pipeline(self, data_input, data_bm):
        gray_input = self.RGB2gray(data_input)
        normalized_input = self.normalize(gray_input)
        tiles_input = self.divide_to_tiles(normalized_input)
        good_gamma_input = self.fix_gamma(tiles_input, 1.2)
        reduced_range_input = self.reduce_range(good_gamma_input)
        reduced_range_masks = self.reduce_range(data_bm)
        no_bottom_top_input = self.cut_bottom_top(reduced_range_input)
        no_bottom_top_bm = self.cut_bottom_top(reduced_range_masks)
        input_patches, bm_patches = self.rand_extract_patches(no_bottom_top_input,
                                                        no_bottom_top_bm,
                                                        64,
                                                        64,
                                                        200000,
                                                        'True')
        # np.save('train_input_patches', self.input_patches)
        # np.save('train_bm_patches', self.bm_patches)
        return input_patches, bm_patches

    def RGB2gray(self, data):
        assert (len(data.shape) == 4)  # 4D arrays
        assert (data.shape[1] == 3)
        black_white_images = data[:, 0, :, :] * 0.299 + data[:, 1, :, :] * 0.587 + data[:, 2, :, :] * 0.114
        black_white_images = np.reshape(black_white_images,
                                        (data.shape[0],
                                         1,
                                         data.shape[2],
                                         data.shape[3]))
        return black_white_images

    def normalize(self, data):
        normal_data = np.empty(data.shape)
        data_std = np.std(data)
        data_mean = np.mean(data)
        normal_data = (data - data_mean)/data_std
        for i in range(data.shape[0]):
            normal_data[i] = ((normal_data[i] - np.min(normal_data[i])) / (np.max(normal_data[i])-np.min(normal_data[i])))*255
        return normal_data

    def divide_to_tiles(self, data):
        CLAHE_object = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        divided_data = np.empty(data.shape)
        for i in range(data.shape[0]):
            divided_data[i, 0] = CLAHE_object.apply(np.array(data[i, 0], dtype=np.uint8))
        return divided_data

    def fix_gamma(self, data, gamma=1.0):
        inverted_gamma = 1.0 / gamma
        gammas = np.array([((i / 255.0) ** inverted_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected_data = np.empty(data.shape)
        for i in range(data.shape[0]):
            corrected_data[i, 0] = cv2.LUT(np.array(data[i, 0], dtype=np.uint8), gammas)
        return corrected_data

    def reduce_range(self, data):
        return data/255

    def cut_bottom_top(self, data):
        return data[:,:,9:574,:]

    def is_fully_contained(self, x, y, width, height, patch_height):
        x_ = x - int(width / 2)  # origin (0,0) shifted to image center
        y_ = y - int(height / 2)  # origin (0,0) shifted to image center
        in_radius = 270 - int(patch_height * np.sqrt(
            2.0) / 2.0)  # radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
        radius = np.sqrt((x_ * x_) + (y_ * y_))
        if radius < in_radius:
            return True
        else:
            return False


    def rand_extract_patches(self, data, data_bm, patch_width, patch_height, n_subs, FOV):
        input_patches = np.empty((n_subs, data.shape[1], patch_height, patch_width))
        bm_patches = np.empty((n_subs, data_bm.shape[1], patch_height, patch_width))
        height = data.shape[2]  # height of the full image
        width = data.shape[3]  # width of the full image
        # (0,0) in the center of the image
        patch_per_img = int(n_subs / data.shape[0])  # N_patches equally divided in the full images
        print("patches per full image: " + str(patch_per_img))
        n_iter = 0  # iter over the total numbe rof patches (N_patches)
        for i in range(data.shape[0]):  # loop over the full images
            n_iter_image = 0
            while n_iter_image < patch_per_img:
                x = random.randint(0 + int(patch_width / 2), width - int(patch_width / 2))
                # print "x_center " +str(x_center)
                y = random.randint(0 + int(patch_height / 2), height - int(patch_height / 2))
                # print "y_center " +str(y_center)
                # check whether the patch is fully contained in the FOV
                if FOV == True:
                    if self.is_fully_contained(x, y, width, height, patch_height) == False:
                        continue
                patch = data[i, :, y - int(patch_height / 2):y + int(patch_height / 2),
                        x - int(patch_width / 2):x + int(patch_width / 2)]
                patch_mask = data_bm[i, :, y - int(patch_height / 2):y + int(patch_height / 2),
                             x - int(patch_width / 2):x + int(patch_width / 2)]
                input_patches[n_iter] = patch
                bm_patches[n_iter] = patch_mask
                n_iter += 1  # total
                n_iter_image += 1  # per full_img
        return input_patches, bm_patches