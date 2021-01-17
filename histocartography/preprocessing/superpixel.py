"""This module handles everything related to superpixels"""

import logging
import math
from abc import abstractmethod

import cv2
import numpy as np
from skimage import color, filters
from skimage.color.colorconv import rgb2hed
from skimage.future import graph
from skimage.segmentation import slic

from .pipeline import PipelineStep


class SuperpixelExtractor(PipelineStep):
    """Helper class to extract superpixels from images"""

    def __init__(self, downsampling_factor: int = 1, **kwargs) -> None:
        """Abstract class that extracts superpixels from RGB Images

        Args:
            nr_superpixels (int): Upper bound of super pixels
            downsampling_factor (int, optional): Downsampling factor from the input image
                                                 resolution. Defaults to 1.
        """
        self.downsampling_factor = downsampling_factor
        super().__init__(**kwargs)

    def process(self, input_image: np.ndarray) -> np.ndarray:
        """Return the superpixels of a given input image

        Args:
            input_image (np.array): Input image

        Returns:
            np.array: Extracted superpixels
        """
        logging.debug("Input size: %s", input_image.shape)
        original_height, original_width, _ = input_image.shape
        if self.downsampling_factor != 1:
            input_image = self._downsample(input_image, self.downsampling_factor)
            logging.debug("Downsampled to %s", input_image.shape)
        superpixels = self._extract_superpixels(input_image)
        if self.downsampling_factor != 1:
            superpixels = self._upsample(superpixels, original_height, original_width)
            logging.debug("Upsampled to %s", superpixels.shape)
        return superpixels

    @abstractmethod
    def _extract_superpixels(self, image: np.ndarray) -> np.ndarray:
        """Perform the superpixel extraction

        Args:
            image (np.array): Input tensor

        Returns:
            np.array: Output tensor
        """

    @staticmethod
    def _downsample(image: np.ndarray, downsampling_factor: int) -> np.ndarray:
        """Downsample an input image with a given downsampling factor

        Args:
            image (np.array): Input tensor
            downsampling_factor (int): Factor to downsample

        Returns:
            np.array: Output tensor
        """
        height, width = image.shape[0], image.shape[1]
        new_height = math.floor(height / downsampling_factor)
        new_width = math.floor(width / downsampling_factor)
        downsampled_image = cv2.resize(
            image, (new_height, new_width), interpolation=cv2.INTER_NEAREST
        )
        return downsampled_image

    @staticmethod
    def _upsample(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
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
        nr_superpixels: int,
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
        self.nr_superpixels = nr_superpixels
        self.blur_kernel_size = blur_kernel_size
        self.max_iter = max_iter
        self.compactness = compactness
        self.color_space = color_space
        super().__init__(**kwargs)

    def _extract_superpixels(self, image: np.ndarray) -> np.ndarray:
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
        superpixels += 1  # Handle regionprops that ignores all values of 0
        return superpixels


class SuperpixelMerger(SuperpixelExtractor):
    def __init__(
        self, downsampling_factor: int, threshold: float = 0.06, **kwargs
    ) -> None:
        self.threshold = threshold
        super().__init__(downsampling_factor=downsampling_factor, **kwargs)

    def process(self, input_image: np.ndarray, superpixels: np.ndarray) -> np.ndarray:
        logging.debug("Input size: %s", input_image.shape)
        original_height, original_width, _ = input_image.shape
        if self.downsampling_factor != 1:
            input_image = self._downsample(input_image, self.downsampling_factor)
            superpixels = self._downsample(superpixels, self.downsampling_factor)
            logging.debug("Downsampled to %s", input_image.shape)
        merged_superpixels = self._extract_superpixels(input_image, superpixels)
        if self.downsampling_factor != 1:
            merged_superpixels = self._upsample(
                merged_superpixels, original_height, original_width
            )
            logging.debug("Upsampled to %s", merged_superpixels.shape)
        return merged_superpixels

    def _extract_superpixels(self, input_image, superpixels):
        edges = filters.sobel(color.rgb2gray(input_image))
        g = graph.rag_boundary(superpixels, edges)
        merged_superpixels = graph.merge_hierarchical(
            superpixels,
            g,
            thresh=self.threshold,
            rag_copy=False,
            in_place_merge=True,
            merge_func=self._merge_boundary,
            weight_func=self._weight_boundary,
        )
        merged_superpixels += 1  # Handle regionprops that ignores all values of 0
        return merged_superpixels

    @staticmethod
    def _weight_boundary(graph, src, dst, n):
        """
        Handle merging of nodes of a region boundary region adjacency graph.

        This function computes the `"weight"` and the count `"count"`
        attributes of the edge between `n` and the node formed after
        merging `src` and `dst`.


        Parameters
        ----------
        graph : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        n : int
            A neighbor of `src` or `dst` or both.

        Returns
        -------
        data : dict
            A dictionary with the "weight" and "count" attributes to be
            assigned for the merged node.

        """
        default = {"weight": 0.0, "count": 0}

        count_src = graph[src].get(n, default)["count"]
        count_dst = graph[dst].get(n, default)["count"]

        weight_src = graph[src].get(n, default)["weight"]
        weight_dst = graph[dst].get(n, default)["weight"]

        count = count_src + count_dst
        return {
            "count": count,
            "weight": (count_src * weight_src + count_dst * weight_dst) / count,
        }

    @staticmethod
    def _merge_boundary(graph, src, dst):
        """Call back called before merging 2 nodes.

        In this case we don't need to do any computation here.
        """
        pass
