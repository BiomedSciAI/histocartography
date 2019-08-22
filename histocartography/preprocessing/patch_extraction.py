"""Whole Slide Image Patch Extraction module."""
import logging
import sys
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw

from histocartography.preprocessing.tissue_mask import get_tissue_mask

# from PIL import Image, ImageDraw

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::PREPROCESING::PATCH EXTRACTION')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
h1.setFormatter(formatter)
log.addHandler(h1)


def get_patches(image_id,
                image=None,
                patch_size=None,
                visualize=1,
                image_tissue_mask=None):
    """For generating patches from an already loaded image from WSI stack

    Parameters
    ----------
    image : numpy array
        The image loaded in numpy array

    Returns
    -------
    Numpy Array mask
        The tissue mask of the input image, dimensions same as that of image
    """
    downsample_factor = 4
    downsample_image = cv2.resize(image,
                                  (int(image.shape[1] / downsample_factor),
                                   int(image.shape[0] / downsample_factor)))
    log.debug('image shape: {}'.format(image.shape))
    log.debug('downsample_image shape: {}'.format(downsample_image.shape))
    cv2.imwrite(
        os.path.join(str(image_id) + '_downsample_image.png'),
        downsample_image)

    x_0 = 0
    y_0 = 0
    stride = patch_size
    labels = np.array([0, 1])  # 0 for bg and 1 for tissue region
    patch_counter = 0
    patch_info_coordinates = []
    patch_info_coordinates_visualize = []

    if image_tissue_mask is None:
        image_tissue_mask = get_tissue_mask(
            image)  # pixel value of 255 where tissue present
    binary_tissue_mask = np.zeros((image_tissue_mask.shape), np.uint8)
    binary_tissue_mask[image_tissue_mask == 255] = labels[
        1]  # pixel value of 1 where tissue present

    while ((x_0 + patch_size) <= image.shape[0]):
        while ((y_0 + patch_size) <= image.shape[1]):

            patch_mask = binary_tissue_mask[x_0:x_0 + patch_size, y_0:y_0 +
                                            patch_size]

            tissue_percent_in_patch = (
                np.count_nonzero(patch_mask == labels[1]) * 100) / (
                    patch_size**2)

            if tissue_percent_in_patch >= 50:  # min50 % of area contains tissue
                patch_counter += 1
                patch_info_coordinates.append([
                    patch_counter, x_0, y_0, x_0 + patch_size, y_0 + patch_size
                ])
                patch_info_coordinates_visualize.append([
                    patch_counter,
                    int(y_0 / downsample_factor),
                    int(x_0 / downsample_factor),
                    int((y_0 + patch_size) / downsample_factor),
                    int((x_0 + patch_size) / downsample_factor)
                ])

            y_0 += stride
        x_0 += stride
        y_0 = 0

    log.debug('Total number of patches: {}'.format(
        len(patch_info_coordinates)))
    log.debug('patches extracted')

    # Visualization
    if (visualize == 1):
        box_color = (0, 0, 255, 256)

        Image_ = downsample_image
        # Stack.read_region((0, 0), 3, (size_3[0], size_3[1])) # 5x
        Image_ = Image.fromarray(Image_)
        draw0 = ImageDraw.Draw(Image_, 'RGBA')

        for k, val in enumerate(patch_info_coordinates_visualize):
            draw0.rectangle(
                val[1:],
                outline=box_color)  # , fill=box_color)#(255, 0, 0, 256))

        Image_ = Image_.convert('RGB')
        Image_ = np.asarray(Image_)
        cv2.imwrite(
            os.path.join(str(image_id) + '_patches_outlines.png'), Image_)

    return patch_info_coordinates
