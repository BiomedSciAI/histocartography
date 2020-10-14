from abc import abstractmethod

import cv2
import numpy as np
import torch
from scipy.stats import skew
from skimage.feature import greycomatrix, greycoprops
from skimage.filters.rank import entropy as Entropy
from skimage.measure import regionprops
from skimage.morphology import disk

from utils import PipelineStep


class FeatureExtractor(PipelineStep):
    def __init__(self, **kwargs) -> None:
        """Abstract class that extracts features from superpixels"""
        super().__init__(**kwargs)

    def process(self, input_image: np.array, superpixels: np.array) -> torch.Tensor:
        """Extract features from the input_image for the defined superpixels

        Args:
            input_image (np.array): Original RGB image
            superpixels (np.array): Extracted superpixels

        Returns:
            torch.Tensor: Extracted features
        """
        return self._extract_features(input_image, superpixels)

    @abstractmethod
    def _extract_features(
        self, input_image: np.array, superpixels: np.array
    ) -> torch.Tensor:
        """Extract features from the input_image for the defined superpixels

        Args:
            input_image (np.array): Original RGB image
            superpixels (np.array): Extracted superpixels

        Returns:
            torch.Tensor: Extracted features
        """


class HandcraftedFeatureExtractor(FeatureExtractor):
    """Helper class to extract handcrafted features from superpixels"""

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Extract handcrafted features from images"""
        super().__init__(**kwargs)

    def _extract_features(
        self, input_image: np.array, superpixels: np.array
    ) -> torch.Tensor:
        """Extract handcrafted features from the input_image in the defined superpixel regions

        Args:
            input_image (np.array): Original RGB Image
            superpixels (np.array): Extracted superpixels

        Returns:
            torch.Tensor: Extracted features
        """
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
