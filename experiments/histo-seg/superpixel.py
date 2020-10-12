import logging
import math
"""This module handles everything related to superpixels"""

from typing import Union

import torch
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
from skimage.feature import greycomatrix, greycoprops
from skimage.filters.rank import entropy as Entropy
from skimage.measure import regionprops
from skimage.morphology import disk
from skimage.segmentation import mark_boundaries, slic

from constants import EMBEDDINGS_KEY, SUPERPIXEL_KEY


class SuperpixelExtractor:
    """Helper class to extract superpixels from images
    """
    def __init__(
        self,
        nr_superpixels: int,
        downsampling_factor: int = 1,
        blur_kernel_size: float = 0,
    ) -> None:
        """Extracts superpixels from RGB Images

        Args:
            nr_superpixels (int): Upper bound of super pixels
            downsampling_factor (int, optional): Downsampling factor from the input image resolution. Defaults to 1.
            blur_kernel_size (float, optional): Size of Gaussian smoothing kernel. Defaults to 0.
        """
        self.downsampling_factor = downsampling_factor
        self.blur_kernel_size = blur_kernel_size
        self.nr_superpixels = nr_superpixels

    def process(self, input_image: np.array) -> np.array:
        """Return the superpixels of a given input image

        Args:
            input_image (np.array): Input image

        Returns:
            np.array: Extracted superpixels
        """
        superpixels = self._extract_superpixels(input_image)
        return superpixels

    def process_and_save(self, input_image: np.array, output_path: str) -> str:
        """Calculate the superpixels of a given input image and save in the provided path as as .h5 file

        Args:
            input_image (np.array): Input image
            output_path (str): Output path

        Returns:
            str: Name of the dataset of the file in output_path
        """
        superpixels = self.process(input_image)
        output_file = h5py.File(output_path, "w")
        output_file.create_dataset(SUPERPIXEL_KEY, data=superpixels, dtype="float32")
        output_file.close()

    def _extract_superpixels(self, image: np.array) -> np.array:
        """Perform the superpixel extraction

        Args:
            image (np.array): Input tensor

        Returns:
            np.array: Output tensor
        """
        logging.debug(f"Input size: {image.shape}")
        original_height, original_width, _ = image.shape
        if self.downsampling_factor != 1:
            image = self._downsample(image, self.downsampling_factor)
            logging.debug(f"Downsampled to {image.shape}")
        superpixels = slic(
            image,
            sigma=self.blur_kernel_size,
            n_segments=self.nr_superpixels,
            max_iter=10,
            compactness=20,
            start_label=1,
        )
        if self.downsampling_factor != 1:
            superpixels = self._upsample(superpixels, original_height, original_width)
            logging.debug(f"Upsampled to {superpixels.shape}")
        return superpixels

    @staticmethod
    def _downsample(image: np.array, downsampling_factor: int) -> np.array:
        """Downsample an input image with a given downsampling factor

        Args:
            image (np.array): Input tensor
            downsampling_factor (int): Factor to downsample

        Returns:
            np.array: Output tensor
        """
        height, width, _ = image.shape
        new_height = math.floor(height / downsampling_factor)
        new_width = math.floor(width / downsampling_factor)
        downsampled_image = cv2.resize(
            image, (new_height, new_width), interpolation=cv2.INTER_NEAREST
        )
        return downsampled_image

    @staticmethod
    def _upsample(image: np.array, new_height: int, new_width: int) -> np.array:
        """Upsample an input image to a speficied new height and width

        Args:
            image (np.array): Input tensor
            new_height (int): Target height
            new_width (int): Target width

        Returns:
            np.array: Output tensor
        """
        upsampled_image = cv2.resize(
            image, (new_height, new_width), interpolation=cv2.INTER_NEAREST
        )
        return upsampled_image


class SuperpixelVisualizer:
    """Helper class that handles visualizing superpixels in a notebook
    """
    def __init__(
        self, height: int = 14, width: int = 14, patch_size: int = 1000
    ) -> None:
        """Helper class to display the output of superpixel algorithms

        Args:
            height (int, optional): Height of the figure. Defaults to 14.
            width (int, optional): Width of the figure. Defaults to 14.
            patch_size (int, optional): Size of a random patch. Defaults to 1000.
        """
        self.height = height
        self.width = width
        self.patch_size = patch_size

    def show_random_patch(self, image: np.array, superpixels: np.array) -> None:
        """Show a random patch of the given superpixels

        Args:
            image (np.array): Input image
            superpixels (np.array): Input superpixels
        """
        width, height, _ = image.shape
        patch_size = min(width, height, self.patch_size)
        x_lower = np.random.randint(0, width - patch_size)
        x_upper = x_lower + patch_size
        y_lower = np.random.randint(0, height - patch_size)
        y_upper = y_lower + patch_size
        self.show(
            image[x_lower:x_upper, y_lower:y_upper],
            superpixels[x_lower:x_upper, y_lower:y_upper],
        )

    def show(self, image: np.array, superpixels: np.array) -> None:
        """Show the given superpixels overlayed over the image

        Args:
            image (np.array): Input image
            superpixels (np.array): Input superpixels
        """
        fig, ax = plt.subplots(figsize=(self.height, self.width))
        marked_image = mark_boundaries(image, superpixels)
        ax.imshow(marked_image)
        ax.set_axis_off()
        fig.show()


class HandcraftedFeatureExtractor:
    """Helper class to extract handcrafted features from superpixels
    """
    def __init__(self) -> None:
        pass

    def process(self, input_image: np.array, superpixels: np.array) -> torch.Tensor:
        node_feat = []

        img_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        img_square = np.square(input_image)

        # -------------------------------------------------------------------------- Entropy per channel
        img_entropy = Entropy(img_gray, disk(3))

        # For each super-pixel
        regions = regionprops(superpixels)

        for _, region in enumerate(regions):
            sp_mask = np.array(superpixels == region["label"], np.uint8)
            sp_rgb = cv2.bitwise_and(input_image, input_image, mask=sp_mask)
            sp_gray = img_gray * sp_mask
            mask_size = np.sum(sp_mask)
            mask_idx = np.where(sp_mask != 0)

            # -------------------------------------------------------------------------- CONTOUR-BASED SHAPE FEATURES
            # Compute using mask [12 features]
            area = region["area"]
            convex_area = region["convex_area"]
            eccentricity = region["eccentricity"]
            equivalent_diameter = region["equivalent_diameter"]
            euler_number = region["euler_number"]
            extent = region["extent"]
            filled_area = region["filled_area"]
            major_axis_length = region["major_axis_length"]
            minor_axis_length = region["minor_axis_length"]
            orientation = region["orientation"]
            perimeter = region["perimeter"]
            solidity = region["solidity"]
            feats_shape = [
                area,
                convex_area,
                eccentricity,
                equivalent_diameter,
                euler_number,
                extent,
                filled_area,
                major_axis_length,
                minor_axis_length,
                orientation,
                perimeter,
                solidity,
            ]

            # -------------------------------------------------------------------------- COLOR FEATURES
            # (rgb color space) [13 x 3 features]
            def color_features_per_channel(img_rgb_ch, img_rgb_sq_ch):
                codes = img_rgb_ch[mask_idx[0], mask_idx[1]].ravel()
                hist, _ = np.histogram(codes, bins=np.arange(0, 257, 32))  # 8 bins
                feats_ = list(hist / mask_size)
                color_mean = np.mean(codes)
                color_std = np.std(codes)
                color_median = np.median(codes)
                color_skewness = skew(codes)

                codes = img_rgb_sq_ch[mask_idx[0], mask_idx[1]].ravel()
                color_energy = np.mean(codes)

                feats_.append(color_mean)
                feats_.append(color_std)
                feats_.append(color_median)
                feats_.append(color_skewness)
                feats_.append(color_energy)
                return feats_

            # enddef

            feats_r = color_features_per_channel(sp_rgb[:, :, 0], img_square[:, :, 0])
            feats_g = color_features_per_channel(sp_rgb[:, :, 1], img_square[:, :, 1])
            feats_b = color_features_per_channel(sp_rgb[:, :, 2], img_square[:, :, 2])
            feats_color = [feats_r, feats_g, feats_b]
            feats_color = [item for sublist in feats_color for item in sublist]

            # -------------------------------------------------------------------------- TEXTURE FEATURES
            # Entropy (gray color space) [1 feature]
            entropy = cv2.mean(img_entropy, mask=sp_mask)[0]

            # GLCM texture features (gray color space) [5 features]
            glcm = greycomatrix(sp_gray, [1], [0])
            # Filter out the first row and column
            filt_glcm = glcm[1:, 1:, :, :]

            glcm_contrast = greycoprops(filt_glcm, prop="contrast")
            glcm_contrast = glcm_contrast[0, 0]
            glcm_dissimilarity = greycoprops(filt_glcm, prop="dissimilarity")
            glcm_dissimilarity = glcm_dissimilarity[0, 0]
            glcm_homogeneity = greycoprops(filt_glcm, prop="homogeneity")
            glcm_homogeneity = glcm_homogeneity[0, 0]
            glcm_energy = greycoprops(filt_glcm, prop="energy")
            glcm_energy = glcm_energy[0, 0]
            glcm_ASM = greycoprops(filt_glcm, prop="ASM")
            glcm_ASM = glcm_ASM[0, 0]

            feats_texture = [
                entropy,
                glcm_contrast,
                glcm_dissimilarity,
                glcm_homogeneity,
                glcm_energy,
                glcm_ASM,
            ]

            # -------------------------------------------------------------------------- STACKING ALL FEATURES
            sp_feats = feats_shape + feats_color + feats_texture

            features = np.hstack(sp_feats)
            node_feat.append(features)
        # endfor

        node_feat = np.vstack(node_feat)
        return torch.Tensor(node_feat)

    def process_and_save(
        self, input_image: np.array, superpixels: np.array, output_path: str
    ) -> None:
        features = self.process(input_image, superpixels)
        output_file = h5py.File(output_path, "w")
        output_file.create_dataset(EMBEDDINGS_KEY, data=features, dtype="float32")
        output_file.close()
