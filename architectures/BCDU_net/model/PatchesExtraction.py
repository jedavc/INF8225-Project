import random
import math
import numpy as np

class PatchesExtraction():
    def __init__(self):
        pass

    def image_centers(self, width, height, patch_width, patch_height):
        center_x = random.randint(0 + patch_width, width - patch_width)
        center_y = random.randint(0 + patch_height, height - patch_height)
        return center_x, center_y

    def divide_to_patches(self, data, data_bm, image, patch_height, patch_width, x, y):
        patch = data[image, :, y - patch_height:y + patch_height, x - patch_width:x + patch_width]
        patch_mask = data_bm[image, :, y - patch_height:y + patch_height, x - patch_width:x + patch_width]
        return patch, patch_mask

    def rand_extract_patches(self, data, data_bm, patch_width, patch_height, n_subs):
        input_patches = np.empty((n_subs, data.shape[1], patch_height, patch_width))
        bm_patches = np.empty((n_subs, data_bm.shape[1], patch_height, patch_width))
        # Image dimensions
        height = data.shape[2]
        width = data.shape[3]
        n_images = data.shape[0]

        patches_per_img = int(n_subs / n_images)
        print("patches per full image: " + str(patches_per_img))

        n_iter_all = 0  # iter over the total number of patches (N_patches)
        for image in range(n_images):  # loop over the full images
            n_iter_image = 0
            while n_iter_image < patches_per_img:
                half_patch_h = int(patch_height / 2)
                half_patch_w = int(patch_width / 2)
                x, y = self.image_centers(width, height, half_patch_w, half_patch_h)
                if self.check_inside(x, y, width, height, patch_height):
                    patch, patch_mask = self.divide_to_patches(data, data_bm, image, half_patch_h, half_patch_w, x, y)
                    input_patches[n_iter_all] = patch
                    bm_patches[n_iter_all] = patch_mask
                    n_iter_all += 1  # total
                    n_iter_image += 1  # per full_img

                else:
                    continue
        return input_patches, bm_patches

    def check_inside(self, x_center, y_center, width, height, patch_height):
        half_width = int(width / 2)
        half_height = int(height/2)
        x = x_center - half_width  # origin (0,0) shifted to image center
        y = y_center - half_height # origin (0,0) shifted to image center
        patch_diagonal = int(patch_height * np.sqrt(2.0) / 2.0)
        inside = 270 - patch_diagonal # radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
        radius = np.sqrt((x **2) + (y**2))
        if radius < inside:
            return True
        else:
            return False

    def view_patches(self, data, patch_height=64, patch_width=64, stride_height=5, stride_width=5):
        n_images = data.shape[0]
        height = data.shape[2]  # height of the full image
        width = data.shape[3]  # width of the full image
        height_val = math.floor((height - patch_height) / stride_height + 1)
        width_val = math.floor((width - patch_width) / stride_width + 1)
        return self.create_patches(data, n_images, height_val, width_val, patch_height, patch_width, stride_height, stride_width)

    def create_patches(self, data, n_images, height_val, width_val, patch_height, patch_width, stride_height, stride_width):
        n_patches_per_img = height_val * width_val
        n_patches_tot = n_patches_per_img * n_images
        patches = np.empty((n_patches_tot, data.shape[1], patch_height, patch_width))
        image_idx = 0
        for image in range(n_images):  # loop over the full images
            for height_pixel in range(height_val):
                for width_pixel in range(width_val):
                    patch = data[image, :, height_pixel * stride_height:(height_pixel * stride_height) + patch_height,
                            width_pixel * stride_width:(width_pixel * stride_width) + patch_width]
                    patches[image_idx] = patch
                    image_idx += 1  # total
        return patches

    def remove_overlap(self, data, patch_height=64, patch_width=64, stride_height=5, stride_width=5):
        height = data.shape[2]  # height of the full image
        width = data.shape[3]  # width of the full image
        height_leftover = (height - patch_height) % stride_height  # leftover on the h dim
        width_leftover = (width - patch_width) % stride_width  # leftover on the w dim
        # data = self.change_dim(data, height_leftover, height, width, stride_height, 'False')
        data = self.change_dim(data, width_leftover, height, width, stride_width, 'True')
        print("New images shape: \n" + str(data.shape))
        return data

    def change_dim(self, data, leftover, height, width, stride_dim, is_width):
        new_dim_img = None
        if (leftover != 0):

            if is_width:
                adjusted_dim = width + (stride_dim - leftover)
                new_dim_img = np.zeros((data.shape[0], data.shape[1], data.shape[2], adjusted_dim))
                new_dim_img[0:data.shape[0], 0:data.shape[1], 0:data.shape[2], 0:width] = data

            else:
                # TO REMOVE IF NO USE
                adjusted_dim = height + (stride_dim - leftover)
                new_dim_img = np.zeros((data.shape[0], data.shape[1], adjusted_dim, width))
                new_dim_img[0:data.shape[0], 0:data.shape[1], 0:height, 0:width] = data
        return new_dim_img