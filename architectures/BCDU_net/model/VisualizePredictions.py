import numpy as np
import math

from architectures.BCDU_net.model.Preprocessing import Preprocessing


class VisualizePredicitons():
    def __init__(self):
        self.stride_weight = 5
        self.stride_height = 5
        pass


    def build_images(self, predictions, height, width):
        assert (len(predictions.shape) == 4)  # 4D arrays
        assert (predictions.shape[1] == 1 or predictions.shape[1] == 3)
        n_preds = predictions.shape[0]
        patch_height = predictions.shape[2]
        patch_width = predictions.shape[3]
        n_patches_in_height = math.floor((height - patch_height) / self.stride_height + 1)
        n_patches_in_width = math.floor((width - patch_width) / self.stride_weight + 1)
        n_patches = n_patches_in_height * n_patches_in_width
        all_images = math.floor(n_preds/n_patches)
        print ("According to the dimension inserted, there are " +str(all_images) +" full images (of " +str(height)+"x" +str(width) +" each)")
        sum_probas, sum = self.patches_sum_probas(all_images, predictions, height, width, patch_height, patch_width, n_patches_in_height, n_patches_in_width)
        formatted_predictions = sum_probas/sum
        print (formatted_predictions.shape)
        # assert(np.max(formatted_predictions)<=1.0) #max value for a pixel is 1.0
        # assert(np.min(formatted_predictions)>=0.0) #min value for a pixel is 0.0
        return formatted_predictions

    def patches_sum_probas(self, all_images, predictions, height, width, patch_height, patch_width, n_patches_in_height, n_patches_in_width):
        sum_probas = np.zeros((all_images, predictions.shape[1], height, width))  #itialize to zero mega array with sum of Probabilities
        sum = np.zeros((all_images, predictions.shape[1], height, width))
        patches_iter = 0  # iterator over all the patches
        for image in range(all_images):
            for i_height in range(n_patches_in_height):
                for j_width in range(n_patches_in_width):
                    tmp_dim = i_height * self.stride_height
                    tmp_dim2 = j_width * self.stride_weight
                    sum_probas[image, :, tmp_dim:(tmp_dim) + patch_height, tmp_dim2:(tmp_dim2) + patch_width] += predictions[patches_iter]
                    sum[image, :, tmp_dim:(tmp_dim) + patch_height, tmp_dim2:(tmp_dim2) + patch_width] += 1
                    patches_iter += 1
        # assert (patches_iter == predictions.shape[0])
        # assert (np.min(sum) >= 1.0)  # at least one
        return sum_probas, sum

    def set_black_border(self, predictions, border_masks):
        assert (len(predictions.shape) == 4)  # 4D arrays
        assert (predictions.shape[1] == 1 or predictions.shape[1] == 3)  # check the channel is 1 or 3
        height = predictions.shape[2]
        width = predictions.shape[3]
        for image in range(predictions.shape[0]):  # loop over the full images
            for pixel_width in range(width):
                for pixel_height in range(height):
                    if self.in_view(image, pixel_width, pixel_height, border_masks) == False:
                        predictions[image, :, pixel_height, pixel_width] = 0.0

    def in_view(self, image, width_pixel, height_pixel, original_bm):
        assert (len(original_bm.shape)==4)  #4D arrays
        assert (original_bm.shape[1]==1)  #DRIVE masks is black and white
        # DRIVE_masks = DRIVE_masks/255.  #NOOO!! otherwise with float numbers takes forever!!
        orig_height = original_bm.shape[2]
        orig_width = original_bm.shape[3]

        if (width_pixel >= orig_width or height_pixel >= orig_height):
            return False

        if (original_bm[image, 0, height_pixel, width_pixel] > 0):
            return True
        else:
            return False

    def set_original_dimensions(self, data, test_inputs):
        height = test_inputs.shape[2]
        width = test_inputs.shape[3]
        return data[:,:,0:height,0:width]

    def field_fo_view(self, images, masks, original_bm):
        assert (len(images.shape) == 4 and len(masks.shape) == 4)  # 4D arrays
        assert (images.shape[0] == masks.shape[0])
        assert (images.shape[2] == masks.shape[2])
        assert (images.shape[3] == masks.shape[3])
        assert (images.shape[1] == 1 and masks.shape[1] == 1)  # check the channel is 1
        height = images.shape[2]
        width = images.shape[3]
        new_preds = []
        new_masks = []
        for image in range(images.shape[0]):  # loop over the full images
            for width_pixel in range(width):
                for height_pixel in range(height):
                    if self.in_view(image, width_pixel, height_pixel, original_bm):
                        new_preds.append(images[image, :, height_pixel, width_pixel])
                        new_masks.append(masks[image, :, height_pixel, width_pixel])
        return np.asarray(new_preds), np.asarray(new_masks)

    def make_visualizable(self, predictions, new_h, new_w, evaluation, test_prepro_bm):
        prediction_images = self.build_images(predictions, new_h, new_w)
        prepro = Preprocessing()
        images, _ = prepro.run_preprocess_pipeline(evaluation.test_inputs[0:prediction_images.shape[0], :, :, :], "test")
        self.set_black_border(prediction_images, evaluation.test_bm)
        images = self.set_original_dimensions(images, evaluation.test_inputs)
        predictions_images = self.set_original_dimensions(prediction_images, evaluation.test_inputs)
        gt = self.set_original_dimensions(test_prepro_bm, evaluation.test_inputs)
        return images, predictions_images, gt