"""This module handles everything related to superpixels"""

import logging
import math
import warnings
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict

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
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
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
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
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
        self,
        downsampling_factor: int,
        threshold: float = 0.06,
        connectivity: int = 2,
        **kwargs,
    ) -> None:
        """Base superpixel merger

        Args:
            downsampling_factor (int): Downsampling factor to use
            threshold (float, optional): Connectivity threshold. Defaults to 0.06.
            connectivity (int, optional): Connectivity for merging graph. Defaults to 2.
        """
        self.threshold = threshold
        self.connectivity = connectivity
        super().__init__(downsampling_factor=downsampling_factor, **kwargs)

    def process(self, input_image: np.ndarray, superpixels: np.ndarray) -> np.ndarray:
        """Merge superpixels based on an initial superpixel segmentation and the input image

        Args:
            input_image (np.ndarray): Input image
            superpixels (np.ndarray): Initial superpixel segmentation

        Returns:
            np.ndarray: Merged superpixel segmentation
        """
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

    def _extract_superpixels(
        self, input_image: np.ndarray, superpixels: np.ndarray
    ) -> np.ndarray:
        g = self._generate_graph(input_image, superpixels)
        merged_superpixels = graph.merge_hierarchical(
            superpixels,
            g,
            thresh=self.threshold,
            rag_copy=False,
            in_place_merge=True,
            merge_func=self._merging_function,
            weight_func=self._weighting_function,
        )
        merged_superpixels += 1  # Handle regionprops that ignores all values of 0
        return merged_superpixels

    @abstractmethod
    def _generate_graph(
        self, input_image: np.ndarray, superpixels: np.ndarray
    ) -> graph.RAG:
        """Generate a graph based on the input image and initial superpixel segmentation"""

    @abstractmethod
    def _weighting_function(
        self, graph: graph.RAG, src: int, dst: int, n: int
    ) -> Dict[str, Any]:
        """
        Handle merging of nodes of a region boundary region adjacency graph.
        """

    @abstractmethod
    def _merging_function(self, graph: graph.RAG, src: int, dst: int) -> None:
        """Call back called before merging 2 nodes."""


class EdgeSuperpixelMerger(SuperpixelMerger):
    def __init__(self, downsampling_factor: int, **kwargs) -> None:
        """Superpixel merging based on edge features"""
        warnings.warn(
            "The use of EdgeSuperpixelMerger is depreciated. Use EdgeMergedSuperpixelExtractor instead",
            DeprecationWarning,
        )
        super().__init__(downsampling_factor, **kwargs)

    def _generate_graph(
        self, input_image: np.ndarray, superpixels: np.ndarray
    ) -> graph.RAG:
        """Generate a graph based on the input image and initial superpixel segmentation

        Args:
            input_image (np.ndarray): Input image
            superpixels (np.ndarray): Initial superpixel segmentation

        Returns:
            graph.RAG: Connectivity graph
        """
        edges = filters.sobel(color.rgb2gray(input_image))
        return graph.rag_boundary(superpixels, edges, connectivity=self.connectivity)

    @staticmethod
    def _weighting_function(
        graph: graph.RAG, src: int, dst: int, n: int
    ) -> Dict[str, Any]:
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
    def _merging_function(graph: graph.RAG, src: int, dst: int) -> None:
        """Call back called before merging 2 nodes.

        In this case we don't need to do any computation here.
        """
        pass


class SpecialSuperpixelMerger(SuperpixelMerger):
    def __init__(
        self,
        downsampling_factor: int,
        w_hist: float = 0.5,
        w_mean: float = 0.5,
        **kwargs,
    ) -> None:
        """Alternative superpixel merger taken from the HACT-Net Implementation

        Args:
            downsampling_factor (int): Factor to use
            w_hist (float, optional): Weight of the histogram features for merging. Defaults to 0.5.
            w_mean (float, optional): Weight of the mean features for merging. Defaults to 0.5.
        """
        warnings.warn(
            "The use of SpecialSuperpixelMerger is depreciated. Use ColorMergedSuperpixelExtractor instead",
            DeprecationWarning,
        )
        self.w_hist = w_hist
        self.w_mean = w_mean
        super().__init__(downsampling_factor, **kwargs)

    def _color_features_per_channel(self, img_ch: np.ndarray) -> np.ndarray:
        """Extract color histograms from image channel

        Args:
            img_ch (np.ndarray): Image channel

        Returns:
            np.ndarray: Histogram of the image channel
        """
        hist, _ = np.histogram(img_ch, bins=np.arange(0, 257, 64))  # 8 bins
        return hist

    def _generate_graph(
        self, input_image: np.ndarray, superpixels: np.ndarray
    ) -> np.ndarray:
        g = graph.RAG(superpixels, connectivity=self.connectivity)

        for n in g:
            g.nodes[n].update(
                {
                    "labels": [n],
                    "N": 0,
                    "x": np.array([0, 0, 0]),
                    "y": np.array([0, 0, 0]),
                    "r": np.array([]),
                    "g": np.array([]),
                    "b": np.array([]),
                }
            )

        for index in np.ndindex(superpixels.shape):
            current = superpixels[index]
            g.nodes[current]["N"] += 1
            g.nodes[current]["x"] += input_image[index]
            g.nodes[current]["y"] = np.vstack(
                (g.nodes[current]["y"], input_image[index])
            )

        for n in g:
            g.nodes[n]["mean"] = g.nodes[n]["x"] / g.nodes[n]["N"]
            g.nodes[n]["mean"] = g.nodes[n]["mean"] / np.linalg.norm(g.nodes[n]["mean"])

            g.nodes[n]["y"] = np.delete(g.nodes[n]["y"], 0, axis=0)
            g.nodes[n]["r"] = self._color_features_per_channel(g.nodes[n]["y"][:, 0])
            g.nodes[n]["g"] = self._color_features_per_channel(g.nodes[n]["y"][:, 1])
            g.nodes[n]["b"] = self._color_features_per_channel(g.nodes[n]["y"][:, 2])

            g.nodes[n]["r"] = g.nodes[n]["r"] / np.linalg.norm(g.nodes[n]["r"])
            g.nodes[n]["g"] = g.nodes[n]["r"] / np.linalg.norm(g.nodes[n]["g"])
            g.nodes[n]["b"] = g.nodes[n]["r"] / np.linalg.norm(g.nodes[n]["b"])

        for x, y, d in g.edges(data=True):
            diff_mean = np.linalg.norm(g.nodes[x]["mean"] - g.nodes[y]["mean"]) / 2

            diff_r = np.linalg.norm(g.nodes[x]["r"] - g.nodes[y]["r"]) / 2
            diff_g = np.linalg.norm(g.nodes[x]["g"] - g.nodes[y]["g"]) / 2
            diff_b = np.linalg.norm(g.nodes[x]["b"] - g.nodes[y]["b"]) / 2
            diff_hist = (diff_r + diff_g + diff_b) / 3

            diff = self.w_hist * diff_hist + self.w_mean * diff_mean

            d["weight"] = diff

        return g

    def _weighting_function(
        self, graph: graph.RAG, src: int, dst: int, n: int
    ) -> Dict[str, Any]:
        diff_mean = np.linalg.norm(graph.nodes[dst]["mean"] - graph.nodes[n]["mean"])

        diff_r = np.linalg.norm(graph.nodes[dst]["r"] - graph.nodes[n]["r"]) / 2
        diff_g = np.linalg.norm(graph.nodes[dst]["g"] - graph.nodes[n]["g"]) / 2
        diff_b = np.linalg.norm(graph.nodes[dst]["b"] - graph.nodes[n]["b"]) / 2
        diff_hist = (diff_r + diff_g + diff_b) / 3

        diff = self.w_hist * diff_hist + self.w_mean * diff_mean

        return {"weight": diff}

    def _merging_function(self, graph: graph.RAG, src: int, dst: int) -> None:
        graph.nodes[dst]["x"] += graph.nodes[src]["x"]
        graph.nodes[dst]["N"] += graph.nodes[src]["N"]
        graph.nodes[dst]["mean"] = graph.nodes[dst]["x"] / graph.nodes[dst]["N"]
        graph.nodes[dst]["mean"] = graph.nodes[dst]["mean"] / np.linalg.norm(
            graph.nodes[dst]["mean"]
        )

        graph.nodes[dst]["y"] = np.vstack(
            (graph.nodes[dst]["y"], graph.nodes[src]["y"])
        )
        graph.nodes[dst]["r"] = self._color_features_per_channel(
            graph.nodes[dst]["y"][:, 0]
        )
        graph.nodes[dst]["g"] = self._color_features_per_channel(
            graph.nodes[dst]["y"][:, 1]
        )
        graph.nodes[dst]["b"] = self._color_features_per_channel(
            graph.nodes[dst]["y"][:, 2]
        )

        graph.nodes[dst]["r"] = graph.nodes[dst]["r"] / np.linalg.norm(
            graph.nodes[dst]["r"]
        )
        graph.nodes[dst]["g"] = graph.nodes[dst]["r"] / np.linalg.norm(
            graph.nodes[dst]["g"]
        )
        graph.nodes[dst]["b"] = graph.nodes[dst]["r"] / np.linalg.norm(
            graph.nodes[dst]["b"]
        )


class MergedSuperpixelExtractor(SuperpixelExtractor):
    def __init__(
        self,
        nr_superpixels: int,
        blur_kernel_size: float = 0,
        compactness: int = 30,
        threshold: float = 0.06,
        connectivity: int = 2,
        **kwargs,
    ) -> None:
        """Extract superpixels with the SLIC algorithm

        Args:
            blur_kernel_size (float, optional): Size of the blur kernel. Defaults to 0.
            max_iter (int, optional): Number of iterations of the slic algorithm. Defaults to 10.
            compactness (int, optional): Compactness of the superpixels. Defaults to 30.
            threshold (float, optional): Connectivity threshold. Defaults to 0.06.
            connectivity (int, optional): Connectivity for merging graph. Defaults to 2.
        """
        self.nr_superpixels = nr_superpixels
        self.blur_kernel_size = blur_kernel_size
        self.compactness = compactness
        self.threshold = threshold
        self.connectivity = connectivity
        super().__init__(**kwargs)

    def _extract_initial_superpixels(self, image: np.ndarray) -> np.ndarray:
        superpixels = slic(
            image,
            sigma=self.blur_kernel_size,
            n_segments=self.nr_superpixels,
            compactness=self.compactness,
        )
        superpixels += 1  # Handle regionprops that ignores all values of 0
        return superpixels

    def _merge_superpixels(
        self, input_image: np.ndarray, initial_superpixels: np.ndarray
    ) -> np.ndarray:
        g = self._generate_graph(input_image, initial_superpixels)
        merged_superpixels = graph.merge_hierarchical(
            initial_superpixels,
            g,
            thresh=self.threshold,
            rag_copy=False,
            in_place_merge=True,
            merge_func=self._merging_function,
            weight_func=self._weighting_function,
        )
        merged_superpixels += 1  # Handle regionprops that ignores all values of 0
        return merged_superpixels

    @abstractmethod
    def _generate_graph(
        self, input_image: np.ndarray, superpixels: np.ndarray
    ) -> graph.RAG:
        """Generate a graph based on the input image and initial superpixel segmentation"""

    @abstractmethod
    def _weighting_function(
        self, graph: graph.RAG, src: int, dst: int, n: int
    ) -> Dict[str, Any]:
        """
        Handle merging of nodes of a region boundary region adjacency graph.
        """

    @abstractmethod
    def _merging_function(self, graph: graph.RAG, src: int, dst: int) -> None:
        """Call back called before merging 2 nodes."""

    @staticmethod
    def _check_translator_consistency(
        instance_map, merged_instance_map, translator
    ) -> None:
        for instance_id in np.unique(merged_instance_map):
            assert (
                instance_id in translator
            ), f"Merged instance id {instance_id} is not mapped to any superpixel: {translator}"
            assert (
                len(translator[instance_id]) > 0
            ), f"Merged instance id {instance_id} is not mapped to any superpixel: {translator}"
        all_values = np.concatenate(list(translator.values()))
        assert len(all_values) == len(
            set(all_values)
        ), f"Mapped values contain duplicates: {all_values}"
        all_values = set(all_values)
        for instance_id in np.unique(instance_map):
            assert (
                instance_id in all_values
            ), f"Inital instance id {instance_id} does not appear in translator"

    def _get_translator(
        self, instance_map: np.ndarray, merged_instance_map: np.ndarray
    ) -> Dict[int, int]:
        """Calculate which instances of the initial instance map belong to each instance of the merged instance map

        Args:
            instance_map (np.ndarray): Initial instance map
            merged_instance_map (np.ndarray): Merged instance map

        Returns:
            Dict[int, int]: Mapping from merged instance map id to initial instance map id
        """
        nr_spx = instance_map.max() + 1
        translator = defaultdict(list)
        for i in range(1, nr_spx):
            mask = instance_map == i
            assignments, counts = np.unique(
                merged_instance_map[mask], return_counts=True
            )
            assignment = assignments[counts.argmax()]
            translator[assignment].append(i)
        self._check_translator_consistency(
            instance_map, merged_instance_map, translator
        )
        return {k: np.array(v) for k, v in translator.items()}

    def _extract_superpixels(self, image: np.ndarray) -> np.ndarray:
        initial_superpixels = self._extract_initial_superpixels(image)
        merged_superpixels = self._merge_superpixels(image, initial_superpixels)
        translator = self._get_translator(initial_superpixels, merged_superpixels)
        self._check_translator_consistency(
            initial_superpixels, merged_superpixels, translator
        )
        return merged_superpixels, initial_superpixels, translator

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
        merged_superpixles, initial_superpixels, mapping = self._extract_superpixels(
            input_image
        )
        if self.downsampling_factor != 1:
            merged_superpixles = self._upsample(
                merged_superpixles, original_height, original_width
            )
            initial_superpixels = self._upsample(
                initial_superpixels, original_height, original_width
            )
            logging.debug("Upsampled to %s", merged_superpixles.shape)
        return merged_superpixles, initial_superpixels, mapping


class EdgeMergedSuperpixelExtractor(MergedSuperpixelExtractor):
    def _generate_graph(
        self, input_image: np.ndarray, superpixels: np.ndarray
    ) -> graph.RAG:
        """Generate a graph based on the input image and initial superpixel segmentation

        Args:
            input_image (np.ndarray): Input image
            superpixels (np.ndarray): Initial superpixel segmentation

        Returns:
            graph.RAG: Connectivity graph
        """
        edges = filters.sobel(color.rgb2gray(input_image))
        return graph.rag_boundary(superpixels, edges, connectivity=self.connectivity)

    @staticmethod
    def _weighting_function(
        graph: graph.RAG, src: int, dst: int, n: int
    ) -> Dict[str, Any]:
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
    def _merging_function(graph: graph.RAG, src: int, dst: int) -> None:
        """Call back called before merging 2 nodes.

        In this case we don't need to do any computation here.
        """
        pass


class ColorMergedSuperpixelExtractor(MergedSuperpixelExtractor):
    def __init__(self, w_hist: float = 0.5, w_mean: float = 0.5, **kwargs) -> None:
        """Superpixel merger based on color attibutes taken from the HACT-Net Implementation

        Args:
            w_hist (float, optional): Weight of the histogram features for merging. Defaults to 0.5.
            w_mean (float, optional): Weight of the mean features for merging. Defaults to 0.5.
        """
        self.w_hist = w_hist
        self.w_mean = w_mean
        super().__init__(**kwargs)

    def _color_features_per_channel(self, img_ch: np.ndarray) -> np.ndarray:
        """Extract color histograms from image channel

        Args:
            img_ch (np.ndarray): Image channel

        Returns:
            np.ndarray: Histogram of the image channel
        """
        hist, _ = np.histogram(img_ch, bins=np.arange(0, 257, 64))  # 8 bins
        return hist

    def _generate_graph(
        self, input_image: np.ndarray, superpixels: np.ndarray
    ) -> np.ndarray:
        g = graph.RAG(superpixels, connectivity=self.connectivity)

        for n in g:
            g.nodes[n].update(
                {
                    "labels": [n],
                    "N": 0,
                    "x": np.array([0, 0, 0]),
                    "y": np.array([0, 0, 0]),
                    "r": np.array([]),
                    "g": np.array([]),
                    "b": np.array([]),
                }
            )

        for index in np.ndindex(superpixels.shape):
            current = superpixels[index]
            g.nodes[current]["N"] += 1
            g.nodes[current]["x"] += input_image[index]
            g.nodes[current]["y"] = np.vstack(
                (g.nodes[current]["y"], input_image[index])
            )

        for n in g:
            g.nodes[n]["mean"] = g.nodes[n]["x"] / g.nodes[n]["N"]
            g.nodes[n]["mean"] = g.nodes[n]["mean"] / np.linalg.norm(g.nodes[n]["mean"])

            g.nodes[n]["y"] = np.delete(g.nodes[n]["y"], 0, axis=0)
            g.nodes[n]["r"] = self._color_features_per_channel(g.nodes[n]["y"][:, 0])
            g.nodes[n]["g"] = self._color_features_per_channel(g.nodes[n]["y"][:, 1])
            g.nodes[n]["b"] = self._color_features_per_channel(g.nodes[n]["y"][:, 2])

            g.nodes[n]["r"] = g.nodes[n]["r"] / np.linalg.norm(g.nodes[n]["r"])
            g.nodes[n]["g"] = g.nodes[n]["r"] / np.linalg.norm(g.nodes[n]["g"])
            g.nodes[n]["b"] = g.nodes[n]["r"] / np.linalg.norm(g.nodes[n]["b"])

        for x, y, d in g.edges(data=True):
            diff_mean = np.linalg.norm(g.nodes[x]["mean"] - g.nodes[y]["mean"]) / 2

            diff_r = np.linalg.norm(g.nodes[x]["r"] - g.nodes[y]["r"]) / 2
            diff_g = np.linalg.norm(g.nodes[x]["g"] - g.nodes[y]["g"]) / 2
            diff_b = np.linalg.norm(g.nodes[x]["b"] - g.nodes[y]["b"]) / 2
            diff_hist = (diff_r + diff_g + diff_b) / 3

            diff = self.w_hist * diff_hist + self.w_mean * diff_mean

            d["weight"] = diff

        return g

    def _weighting_function(
        self, graph: graph.RAG, src: int, dst: int, n: int
    ) -> Dict[str, Any]:
        diff_mean = np.linalg.norm(graph.nodes[dst]["mean"] - graph.nodes[n]["mean"])

        diff_r = np.linalg.norm(graph.nodes[dst]["r"] - graph.nodes[n]["r"]) / 2
        diff_g = np.linalg.norm(graph.nodes[dst]["g"] - graph.nodes[n]["g"]) / 2
        diff_b = np.linalg.norm(graph.nodes[dst]["b"] - graph.nodes[n]["b"]) / 2
        diff_hist = (diff_r + diff_g + diff_b) / 3

        diff = self.w_hist * diff_hist + self.w_mean * diff_mean

        return {"weight": diff}

    def _merging_function(self, graph: graph.RAG, src: int, dst: int) -> None:
        graph.nodes[dst]["x"] += graph.nodes[src]["x"]
        graph.nodes[dst]["N"] += graph.nodes[src]["N"]
        graph.nodes[dst]["mean"] = graph.nodes[dst]["x"] / graph.nodes[dst]["N"]
        graph.nodes[dst]["mean"] = graph.nodes[dst]["mean"] / np.linalg.norm(
            graph.nodes[dst]["mean"]
        )

        graph.nodes[dst]["y"] = np.vstack(
            (graph.nodes[dst]["y"], graph.nodes[src]["y"])
        )
        graph.nodes[dst]["r"] = self._color_features_per_channel(
            graph.nodes[dst]["y"][:, 0]
        )
        graph.nodes[dst]["g"] = self._color_features_per_channel(
            graph.nodes[dst]["y"][:, 1]
        )
        graph.nodes[dst]["b"] = self._color_features_per_channel(
            graph.nodes[dst]["y"][:, 2]
        )

        graph.nodes[dst]["r"] = graph.nodes[dst]["r"] / np.linalg.norm(
            graph.nodes[dst]["r"]
        )
        graph.nodes[dst]["g"] = graph.nodes[dst]["r"] / np.linalg.norm(
            graph.nodes[dst]["g"]
        )
        graph.nodes[dst]["b"] = graph.nodes[dst]["r"] / np.linalg.norm(
            graph.nodes[dst]["b"]
        )
