"""This module handles everything related to superpixels"""

import logging
import math
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import h5py
import numpy as np
from skimage.color.colorconv import rgb2hed
from skimage.future import graph
from skimage.segmentation import slic

from ..pipeline import PipelineStep


class SuperpixelExtractor(PipelineStep):
    """Helper class to extract superpixels from images"""

    def __init__(
        self,
        nr_superpixels: int = None,
        superpixel_size: int = None,
        max_nr_superpixels: Optional[int] = None,
        blur_kernel_size: Optional[float] = 1,
        compactness: Optional[int] = 20,
        max_iterations: Optional[int] = 10,
        threshold: Optional[float] = 0.03,
        connectivity: Optional[int] = 2,
        color_space: Optional[str] = "rgb",
        downsampling_factor: Optional[int] = 1,
        **kwargs,
    ) -> None:
        """Abstract class that extracts superpixels from RGB Images
        Args:
            nr_superpixels (None, int): The number of super pixels before any merging.
            superpixel_size (None, int): The size of super pixels before any merging.
            max_nr_superpixels (int, optional): Upper bound for the number of super pixels.
                                                Useful when providing a superpixel size.
            blur_kernel_size (float, optional): Size of the blur kernel. Defaults to 0.
            compactness (int, optional): Compactness of the superpixels. Defaults to 30.
            max_iterations (int, optional): Number of iterations of the slic algorithm. Defaults to 10.
            threshold (float, optional): Connectivity threshold. Defaults to 0.03.
            connectivity (int, optional): Connectivity for merging graph. Defaults to 2.
            downsampling_factor (int, optional): Downsampling factor from the input image
                                                 resolution. Defaults to 1.
        """
        assert (nr_superpixels is None and superpixel_size is not None) or (
            nr_superpixels is not None and superpixel_size is None
        ), "Provide value for either nr_superpixels or superpixel_size"
        self.nr_superpixels = nr_superpixels
        self.superpixel_size = superpixel_size
        self.max_nr_superpixels = max_nr_superpixels
        self.blur_kernel_size = blur_kernel_size
        self.compactness = compactness
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.connectivity = connectivity
        self.color_space = color_space
        self.downsampling_factor = downsampling_factor
        super().__init__(**kwargs)

    def _process(  # type: ignore[override]
        self, input_image: np.ndarray, tissue_mask: np.ndarray = None
    ) -> np.ndarray:
        """Return the superpixels of a given input image
        Args:
            input_image (np.array): Input image
            tissue_mask (None, np.array): Input tissue mask
        Returns:
            np.array: Extracted superpixels
        """
        logging.debug("Input size: %s", input_image.shape)
        original_height, original_width, _ = input_image.shape
        if self.downsampling_factor != 1:
            input_image = self._downsample(
                input_image, self.downsampling_factor)
            if tissue_mask is not None:
                tissue_mask = self._downsample(
                    tissue_mask, self.downsampling_factor)
            logging.debug("Downsampled to %s", input_image.shape)
        superpixels = self._extract_superpixels(
            image=input_image, tissue_mask=tissue_mask
        )
        if self.downsampling_factor != 1:
            superpixels = self._upsample(
                superpixels, original_height, original_width)
            logging.debug("Upsampled to %s", superpixels.shape)
        return superpixels

    @abstractmethod
    def _extract_superpixels(
        self, image: np.ndarray, tissue_mask: np.ndarray = None
    ) -> np.ndarray:
        """Perform the superpixel extraction
        Args:
            image (np.array): Input tensor
            tissue_mask (np.array): Tissue mask tensor
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
    def _upsample(
            image: np.ndarray,
            new_height: int,
            new_width: int) -> np.ndarray:
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

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Precompute all necessary information

        Args:
            link_path (Union[None, str, Path], optional): Path to link to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save precomputation outputs. Defaults to None.
        """
        if self.save_path is not None and link_path is not None:
            self._link_to_path(Path(link_path) / "superpixels")


class SLICSuperpixelExtractor(SuperpixelExtractor):
    """Use the SLIC algorithm to extract superpixels."""

    def __init__(self, **kwargs) -> None:
        """Extract superpixels with the SLIC algorithm"""
        super().__init__(**kwargs)

    def _get_nr_superpixels(self, image: np.ndarray) -> int:
        """Compute the number of superpixels for initial segmentation
        Args:
            image (np.array): Input tensor
        Returns:
            int: Output number of superpixels
        """
        if self.superpixel_size is not None:
            nr_superpixels = int(
                (image.shape[0] * image.shape[1] / self.superpixel_size)
            )
        elif self.nr_superpixels is not None:
            nr_superpixels = self.nr_superpixels
        if self.max_nr_superpixels is not None:
            nr_superpixels = min(nr_superpixels, self.max_nr_superpixels)
        return nr_superpixels

    def _extract_superpixels(
            self,
            image: np.ndarray,
            *args,
            **kwargs) -> np.ndarray:
        """Perform the superpixel extraction
        Args:
            image (np.array): Input tensor
        Returns:
            np.array: Output tensor
        """
        if self.color_space == "hed":
            image = rgb2hed(image)
        nr_superpixels = self._get_nr_superpixels(image)
        superpixels = slic(
            image,
            sigma=self.blur_kernel_size,
            n_segments=nr_superpixels,
            max_iter=self.max_iterations,
            compactness=self.compactness,
            start_label=1,
        )
        return superpixels


class MergedSuperpixelExtractor(SuperpixelExtractor):
    def __init__(self, **kwargs) -> None:
        """Extract superpixels with the SLIC algorithm"""
        super().__init__(**kwargs)

    def _get_nr_superpixels(self, image: np.ndarray) -> int:
        """Compute the number of superpixels for initial segmentation
        Args:
            image (np.array): Input tensor
        Returns:
            int: Output number of superpixels
        """
        if self.superpixel_size is not None:
            nr_superpixels = int(
                (image.shape[0] * image.shape[1] / self.superpixel_size)
            )
        elif self.nr_superpixels is not None:
            nr_superpixels = self.nr_superpixels
        if self.max_nr_superpixels is not None:
            nr_superpixels = min(nr_superpixels, self.max_nr_superpixels)
        return nr_superpixels

    def _extract_initial_superpixels(self, image: np.ndarray) -> np.ndarray:
        """Extract initial superpixels using SLIC
        Args:
            image (np.array): Input tensor
        Returns:
            np.array: Output tensor
        """
        nr_superpixels = self._get_nr_superpixels(image)
        superpixels = slic(
            image,
            sigma=self.blur_kernel_size,
            n_segments=nr_superpixels,
            compactness=self.compactness,
            max_iter=self.max_iterations,
            start_label=1,
        )
        return superpixels

    def _merge_superpixels(
        self,
        input_image: np.ndarray,
        initial_superpixels: np.ndarray,
        tissue_mask: np.ndarray = None,
    ) -> np.ndarray:
        """Merge the initial superpixels to return merged superpixels
        Args:
            image (np.array): Input image
            initial_superpixels (np.array): Initial superpixels
            tissue_mask (None, np.array): Tissue mask
        Returns:
            np.array: Output merged superpixel tensor
        """
        if tissue_mask is not None:
            # Remove superpixels belonging to background or having < 10% tissue
            # content
            ids_initial = np.unique(initial_superpixels, return_counts=True)
            ids_masked = np.unique(
                tissue_mask * initial_superpixels, return_counts=True
            )

            ctr = 1
            superpixels = np.zeros_like(initial_superpixels)
            for i in range(len(ids_initial[0])):
                id = ids_initial[0][i]
                if id in ids_masked[0]:
                    idx = np.where(id == ids_masked[0])[0]
                    ratio = ids_masked[1][idx] / ids_initial[1][i]
                    if ratio >= 0.1:
                        superpixels[initial_superpixels == id] = ctr
                        ctr += 1

            initial_superpixels = superpixels

        # Merge superpixels within tissue region
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
        mask = np.zeros_like(initial_superpixels)
        mask[initial_superpixels != 0] = 1
        merged_superpixels = merged_superpixels * mask
        return merged_superpixels

    @abstractmethod
    def _generate_graph(
        self, input_image: np.ndarray, superpixels: np.ndarray
    ) -> graph.RAG:
        """Generate a graph based on the input image and initial superpixel segmentation."""

    @abstractmethod
    def _weighting_function(
        self, graph: graph.RAG, src: int, dst: int, n: int
    ) -> Dict[str, Any]:
        """Handle merging of nodes of a region boundary region adjacency graph."""

    @abstractmethod
    def _merging_function(self, graph: graph.RAG, src: int, dst: int) -> None:
        """Call back called before merging 2 nodes."""

    def _extract_superpixels(
        self, image: np.ndarray, tissue_mask: np.ndarray = None
    ) -> np.ndarray:
        """Perform superpixel extraction
        Args:
            image (np.array): Input tensor
            tissue_mask (np.array, optional): Input tissue mask
        Returns:
            np.array: Extracted merged superpixels.
            np.array: Extracted init superpixels, ie before merging.
        """
        initial_superpixels = self._extract_initial_superpixels(image)
        merged_superpixels = self._merge_superpixels(
            image, initial_superpixels, tissue_mask
        )

        return merged_superpixels, initial_superpixels

    def _process(  # type: ignore[override]
        self, input_image: np.ndarray, tissue_mask=None
    ) -> np.ndarray:
        """Return the superpixels of a given input image
        Args:
            input_image (np.array): Input image.
            tissue_mask (None, np.array): Tissue mask.
        Returns:
            np.array: Extracted merged superpixels.
            np.array: Extracted init superpixels, ie before merging.
        """
        logging.debug("Input size: %s", input_image.shape)
        original_height, original_width, _ = input_image.shape
        if self.downsampling_factor is not None and self.downsampling_factor != 1:
            input_image = self._downsample(
                input_image, self.downsampling_factor)
            if tissue_mask is not None:
                tissue_mask = self._downsample(
                    tissue_mask, self.downsampling_factor)
            logging.debug("Downsampled to %s", input_image.shape)
        merged_superpixels, initial_superpixels = self._extract_superpixels(
            input_image, tissue_mask
        )
        if self.downsampling_factor != 1:
            merged_superpixels = self._upsample(
                merged_superpixels, original_height, original_width
            )
            initial_superpixels = self._upsample(
                initial_superpixels, original_height, original_width
            )
            logging.debug("Upsampled to %s", merged_superpixels.shape)
        return merged_superpixels, initial_superpixels

    def _process_and_save(
            self,
            *args: Any,
            output_name: str,
            **kwargs: Any) -> Any:
        """Process and save in the provided path as as .h5 file
        Args:
            output_name (str): Name of output file
        """
        assert (
            self.save_path is not None
        ), "Can only save intermediate output if base_path was not None when constructing the object"
        superpixel_output_path = self.output_dir / f"{output_name}.h5"
        if superpixel_output_path.exists():
            logging.info(
                f"{self.__class__.__name__}: Output of {output_name} already exists, using it instead of recomputing"
            )
            try:
                with h5py.File(superpixel_output_path, "r") as input_file:
                    merged_superpixels, initial_superpixels = self._get_outputs(
                        input_file=input_file)
            except OSError as e:
                print(
                    f"\n\nCould not read from {superpixel_output_path}!\n\n",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"\n\nCould not read from {superpixel_output_path}!\n\n",
                    flush=True)
                raise e
        else:
            merged_superpixels, initial_superpixels = self._process(
                *args, **kwargs)
            try:
                with h5py.File(superpixel_output_path, "w") as output_file:
                    self._set_outputs(
                        output_file=output_file,
                        outputs=(merged_superpixels, initial_superpixels),
                    )
            except OSError as e:
                print(
                    f"\n\nCould not write to {superpixel_output_path}!\n\n",
                    flush=True)
                raise e
        return merged_superpixels, initial_superpixels


class ColorMergedSuperpixelExtractor(MergedSuperpixelExtractor):
    def __init__(
            self,
            w_hist: float = 0.5,
            w_mean: float = 0.5,
            **kwargs) -> None:
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
    ) -> graph:
        """Construct RAG graph using initial superpixel instance map
        Args:
            input_image (np.ndarray): Input image
            superpixels (np.ndarray): Initial superpixel instance map
        Returns:
            graph: Constructed graph
        """
        g = graph.RAG(superpixels, connectivity=self.connectivity)
        if 0 in g.nodes:
            g.remove_node(n=0)  # remove background node

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
            if current == 0:
                continue
            g.nodes[current]["N"] += 1
            g.nodes[current]["x"] += input_image[index]
            g.nodes[current]["y"] = np.vstack(
                (g.nodes[current]["y"], input_image[index])
            )

        for n in g:
            g.nodes[n]["mean"] = g.nodes[n]["x"] / g.nodes[n]["N"]
            g.nodes[n]["mean"] = g.nodes[n]["mean"] / \
                np.linalg.norm(g.nodes[n]["mean"])

            g.nodes[n]["y"] = np.delete(g.nodes[n]["y"], 0, axis=0)
            g.nodes[n]["r"] = self._color_features_per_channel(
                g.nodes[n]["y"][:, 0])
            g.nodes[n]["g"] = self._color_features_per_channel(
                g.nodes[n]["y"][:, 1])
            g.nodes[n]["b"] = self._color_features_per_channel(
                g.nodes[n]["y"][:, 2])

            g.nodes[n]["r"] = g.nodes[n]["r"] / np.linalg.norm(g.nodes[n]["r"])
            g.nodes[n]["g"] = g.nodes[n]["r"] / np.linalg.norm(g.nodes[n]["g"])
            g.nodes[n]["b"] = g.nodes[n]["r"] / np.linalg.norm(g.nodes[n]["b"])

        for x, y, d in g.edges(data=True):
            diff_mean = np.linalg.norm(
                g.nodes[x]["mean"] - g.nodes[y]["mean"]) / 2

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
        diff_mean = np.linalg.norm(
            graph.nodes[dst]["mean"] -
            graph.nodes[n]["mean"])

        diff_r = np.linalg.norm(
            graph.nodes[dst]["r"] - graph.nodes[n]["r"]) / 2
        diff_g = np.linalg.norm(
            graph.nodes[dst]["g"] - graph.nodes[n]["g"]) / 2
        diff_b = np.linalg.norm(
            graph.nodes[dst]["b"] - graph.nodes[n]["b"]) / 2
        diff_hist = (diff_r + diff_g + diff_b) / 3

        diff = self.w_hist * diff_hist + self.w_mean * diff_mean

        return {"weight": diff}

    def _merging_function(self, graph: graph.RAG, src: int, dst: int) -> None:
        graph.nodes[dst]["x"] += graph.nodes[src]["x"]
        graph.nodes[dst]["N"] += graph.nodes[src]["N"]
        graph.nodes[dst]["mean"] = graph.nodes[dst]["x"] / \
            graph.nodes[dst]["N"]
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
