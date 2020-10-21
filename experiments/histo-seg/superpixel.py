"""This module handles everything related to superpixels"""

import logging
import math
from abc import abstractmethod

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color.colorconv import rgb2hed
from skimage.segmentation import mark_boundaries, slic

from utils import PipelineStep


class SuperpixelExtractor(PipelineStep):
    """Helper class to extract superpixels from images"""

    def __init__(
        self, nr_superpixels: int, downsampling_factor: int = 1, **kwargs
    ) -> None:
        """Abstract class that extracts superpixels from RGB Images

        Args:
            nr_superpixels (int): Upper bound of super pixels
            downsampling_factor (int, optional): Downsampling factor from the input image resolution. Defaults to 1.
        """
        self.downsampling_factor = downsampling_factor
        self.nr_superpixels = nr_superpixels
        super().__init__(**kwargs)

    def process(self, input_image: np.ndarray) -> np.array:
        """Return the superpixels of a given input image

        Args:
            input_image (np.array): Input image

        Returns:
            np.array: Extracted superpixels
        """
        logging.debug(f"Input size: {input_image.shape}")
        original_height, original_width, _ = input_image.shape
        if self.downsampling_factor != 1:
            input_image = self._downsample(input_image, self.downsampling_factor)
            logging.debug(f"Downsampled to {input_image.shape}")
        superpixels = self._extract_superpixels(input_image)
        if self.downsampling_factor != 1:
            superpixels = self._upsample(superpixels, original_height, original_width)
            logging.debug(f"Upsampled to {superpixels.shape}")
        return superpixels

    @abstractmethod
    def _extract_superpixels(self, image: np.ndarray) -> np.array:
        """Perform the superpixel extraction

        Args:
            image (np.array): Input tensor

        Returns:
            np.array: Output tensor
        """

    @staticmethod
    def _downsample(image: np.ndarray, downsampling_factor: int) -> np.array:
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
    def _upsample(image: np.ndarray, new_height: int, new_width: int) -> np.array:
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


class SLICSuperpixelExtractor(SuperpixelExtractor):
    """Use the SLIC algorithm to extract superpixels"""

    def __init__(
        self,
        blur_kernel_size: float = 0,
        max_iter: int = 10,
        compactness: int = 30,
        color_space: str = "rgb",
        **kwargs,
    ) -> None:
        """Extract superpixels with the SLIC algorithm

        Args:
            blur_kernel_size (float, optional): Size of the blur kernel. Defaults to 0.
            max_iter (int, optional): Number of iterations of the slic algorithm. Defaults to 10.
            compactness (int, optional): Compactness of the superpixels. Defaults to 30.
        """
        self.blur_kernel_size = blur_kernel_size
        self.max_iter = max_iter
        self.compactness = compactness
        self.color_space = color_space
        super().__init__(**kwargs)

    def _extract_superpixels(self, image: np.ndarray) -> np.array:
        """Perform the superpixel extraction

        Args:
            image (np.array): Input tensor

        Returns:
            np.array: Output tensor
        """
        if self.color_space == "hed":
            image = rgb2hed(image)
        superpixels = slic(
            image,
            sigma=self.blur_kernel_size,
            n_segments=self.nr_superpixels,
            max_iter=self.max_iter,
            compactness=self.compactness,
        )
        superpixels += (
            1  # Handle old version of skimage does not support start_label argument
        )
        return superpixels


class SuperpixelVisualizer:
    """Helper class that handles visualizing superpixels in a notebook"""

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

    def show_random_patch(self, image: np.ndarray, superpixels: np.ndarray) -> None:
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

    def show(self, image: np.ndarray, superpixels: np.ndarray) -> None:
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
