"""Dataloader for precomputed graphs in .bin format"""
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import dgl
import h5py
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs
from histocartography.preprocessing.utils import fast_histogram
from torch.utils.data import Dataset
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
    ToPILImage,
    ToTensor,
)
from tqdm.auto import tqdm

from constants import CENTROID, FEATURES, GNN_NODE_FEAT_IN, LABEL
from utils import read_image


@dataclass
class GraphDatapoint:
    """Dataclass that holds a datapoint for a graph"""

    graph: dgl.DGLGraph
    graph_label: Optional[torch.IntTensor] = None
    node_labels: Optional[torch.IntTensor] = None
    name: Optional[str] = None
    instance_map: Optional[np.ndarray] = None
    segmentation_mask: Optional[np.ndarray] = None
    tissue_mask: Optional[np.ndarray] = None
    additional_segmentation_mask: Optional[np.ndarray] = None

    @property
    def is_weakly_supervised(self):
        return self.graph_label is not None and self.node_labels is None

    @property
    def is_strongly_supervised(self):
        return self.node_labels is not None and self.graph_label is None

    @property
    def can_output_segmentation(self):
        return self.instance_map is not None and self.tissue_mask is not None

    @property
    def has_validation_information(self):
        return self.can_output_segmentation and self.segmentation_mask is not None

    @property
    def has_multiple_annotations(self):
        return (
            self.segmentation_mask is not None
            and self.additional_segmentation_mask is not None
        )


@dataclass
class GraphBatch:
    """Dataclass for a batch of GraphDatapoints"""

    meta_graph: dgl.DGLGraph
    graph_labels: Optional[torch.IntTensor] = None
    node_labels: Optional[torch.IntTensor] = None
    names: Optional[List[str]] = None
    instance_maps: Optional[np.ndarray] = None
    segmentation_masks: Optional[np.ndarray] = None
    tissue_masks: Optional[np.ndarray] = None
    additional_segmentation_masks: Optional[np.ndarray] = None

    @property
    def is_weakly_supervised(self):
        return self.graph_labels is not None and self.node_labels is None

    @property
    def is_strongly_supervised(self):
        return self.node_labels is not None and self.graph_labels is None

    @property
    def can_output_segmentation(self):
        return self.instance_maps is not None and self.tissue_masks is not None

    @property
    def has_validation_information(self):
        return self.can_output_segmentation and self.segmentation_masks is not None

    @property
    def has_multiple_annotations(self):
        return (
            self.segmentation_masks is not None
            and self.additional_segmentation_masks is not None
        )


def collate_graphs(samples: List[GraphDatapoint]) -> GraphBatch:
    """Turns a list of GraphDatapoint into a GraphBatch

    Args:
        samples (List[GraphDatapoint]): Input datapoints

    Returns:
        GraphBatch: Output batch
    """
    merged_datapoints = defaultdict(list)
    for sample in samples:
        for attribute, value in asdict(sample).items():
            if value is not None:
                merged_datapoints[attribute].append(value)

    nr_datapoints = len(samples)
    for attribute, values in merged_datapoints.items():
        assert (
            len(values) == nr_datapoints
        ), f"Could not batch samples, inconsistent attibutes: {samples}"

    def map_name(name: str):
        """Maps names of GraphDatapoint to the names of GraphBatch

        Args:
            name ([str]): Name of GraphDatapoint

        Returns:
            [str]: Name of GraphBatch
        """
        if name == "graph":
            return "meta_graph"
        elif name == "node_labels":
            return "node_labels"
        else:
            return name + "s"

    def merge(name: str, values: Any) -> Any:
        """Merges attribues based on the names

        Args:
            name (str): Name of attibute
            values (Any): Values to merge

        Returns:
            Any: Merged values
        """
        return {
            "graph": dgl.batch,
            "graph_label": torch.stack,
            "node_labels": torch.cat,
        }.get(name, np.stack)(values)

    return GraphBatch(
        **{map_name(k): merge(k, v) for k, v in merged_datapoints.items()}
    )


@dataclass
class ImageDatapoint:
    image: torch.Tensor
    segmentation_mask: Optional[np.ndarray] = None
    tissue_mask: Optional[np.ndarray] = None
    additional_segmentation_mask: Optional[np.ndarray] = None
    name: Optional[str] = None

    @property
    def has_multiple_annotations(self):
        return (
            self.segmentation_mask is not None
            and self.additional_segmentation_mask is not None
        )


class BaseDataset(Dataset):
    def __init__(
        self,
        metadata,
        num_classes,
        background_index,
    ) -> None:
        self._check_metadata(metadata)
        self.metadata = metadata
        self.num_classes = num_classes
        self.background_index = background_index
        self._load(self.metadata)

    @staticmethod
    def _check_metadata(metadata: pd.DataFrame) -> None:
        """Checks that the given metadata has a valid format and all referenced files exist

        Args:
            metadata (pd.DataFrame): Metadata dataframe
        """
        assert (
            not metadata.isna().any().any()
        ), f"Some entries in metadata are NaN: {metadata.isna().any()}"
        assert (
            "width" in metadata and "height" in metadata
        ), f"Metadata lacks image sizes"
        if "graph_path" in metadata:
            for name, row in metadata.iterrows():
                assert (
                    row.graph_path.exists()
                ), f"Graph {name} referenced in metadata does not exist: {row.graph_path}"
        if "superpixel_path" in metadata:
            for name, row in metadata.iterrows():
                assert (
                    row.superpixel_path.exists()
                ), f"Superpixel {name} referenced in metadata does not exist: {row.superpixel_path}"
        if "annotation_path" in metadata:
            for name, row in metadata.iterrows():
                assert (
                    row.annotation_path.exists()
                ), f"Annotation {name} referenced in metadata does not exist: {row.annotation_path}"
        if "processed_image_path" in metadata:
            for name, row in metadata.iterrows():
                assert (
                    row.processed_image_path.exists()
                ), f"Processed Image {name} referenced in metadata does not exist: {row.processed_image_path}"

    def _to_onehot_with_ignore(
        self, input_vector: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Converts an input vector into a one-hot encoded matrix using the num_classes and background class attributes

        Args:
            input_vector (Union[np.ndarray, torch.Tensor]): Input vector (1 dimensional)

        Raises:
            NotImplementedError: Handles only numpy arrays and tensors

        Returns:
            torch.Tensor: One-hot encoded matrix with shape: nr_samples x num_classes
        """
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.Tensor(input_vector.astype(np.int64)).to(torch.int64)
        elif isinstance(input_vector, torch.Tensor):
            input_vector = input_vector.to(torch.int64)
        else:
            raise NotImplementedError(f"Only support numpy arrays and torch tensors")
        one_hot_vector = torch.nn.functional.one_hot(
            input_vector, num_classes=self.num_classes + 1
        )
        clean_one_hot_vector = torch.cat(
            [
                one_hot_vector[:, 0 : self.background_index],
                one_hot_vector[:, self.background_index + 1 :],
            ],
            dim=1,
        )
        return clean_one_hot_vector.to(torch.int8)

    @staticmethod
    def _load_name(i, row):
        return i

    @staticmethod
    def _load_image_size(i, row):
        return (row.height, row.width)

    def _load_datapoint(self, i, row):
        self._image_sizes.append(self._load_image_size(i, row))
        self._names.append(self._load_name(i, row))

    def _initalize_loading(self):
        self._image_sizes = list()
        self._names = list()

    def _finish_loading(self):
        self._image_sizes = np.array(self._image_sizes)
        self._names = np.array(self._names)

    def _load(self, metadata):
        self._initalize_loading()
        for i, row in tqdm(
            metadata.iterrows(), total=len(metadata), desc=f"Dataset Loading"
        ):
            self._load_datapoint(i, row)
        self._finish_loading()

    @property
    def names(self):
        return self._names

    @property
    def image_sizes(self):
        return self._image_sizes


class GraphClassificationDataset(BaseDataset):
    """Dataset used for extracted and dumped graphs"""

    def __init__(
        self,
        tissue_metadata: Optional[pd.DataFrame] = None,
        image_metadata: Optional[pd.DataFrame] = None,
        patch_size: Optional[Tuple[int, int]] = None,
        num_classes: int = 4,
        background_index: int = 4,
        centroid_features: str = "no",
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        return_segmentation_info: bool = False,
        segmentation_downsample_ratio: int = 1,
        image_label_mapper: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        assert centroid_features in [
            "no",
            "cat",
            "only",
        ], f"centroid_features must be in [no, cat, only] but is {centroid_features}"
        assert (
            tissue_metadata is None or "graph_path" in tissue_metadata
        ), f"Metadata lacks graph path ({tissue_metadata.columns})"
        assert (
            image_metadata is None or "graph_path" in image_metadata
        ), f"Metadata lacks graph path ({image_metadata.columns})"
        # Set fields
        self.return_segmentation_info = return_segmentation_info
        self.USE_ANNOTATION2 = (
            tissue_metadata is not None and "annotation2_path" in tissue_metadata
        )
        self.patch_size = patch_size
        self.mean = mean
        self.std = std
        if tissue_metadata is not None:
            self.set_mode("tissue")
        else:
            self.set_mode("image")
        self.downsample = segmentation_downsample_ratio

        # Dataloading
        self._manual_initalize_loading()
        if tissue_metadata is not None:
            super().__init__(tissue_metadata, num_classes, background_index)
            self._tissue_indices = np.arange(0, len(self._graphs))
            if image_metadata is not None:
                self._load(image_metadata)
                self._image_indices = np.arange(0, len(self._graphs))
            else:
                self._image_indices = []
        else:
            super().__init__(image_metadata, num_classes, background_index)
            self._tissue_indices = []
            self._image_indices = np.arange(0, len(self._graphs))
        self._manual_finish_loading()

        # Post processing
        self.name_to_index = dict(zip(self.names, range(len(self.names))))
        self._select_graph_features(centroid_features)
        if image_label_mapper is not None:
            self._graph_labels = self._load_image_level_labels(image_label_mapper)
        else:
            self._graph_labels = self._compute_graph_labels()

    def _initalize_loading(self):
        pass

    def _manual_initalize_loading(self):
        super()._initalize_loading()
        self._graphs = list()
        if self.return_segmentation_info:
            self._superpixels = list()
            self._annotations = list()
            self._tissue_masks = list()
            if self.USE_ANNOTATION2:
                self._annotations2 = list()

    def _finish_loading(self):
        pass

    def _manual_finish_loading(self):
        super()._finish_loading()
        self._graphs = np.array(self._graphs)
        if self.return_segmentation_info:
            self._superpixels = np.array(self._superpixels)
            self._annotations = np.array(self._annotations)
            self._tissue_masks = np.array(self._tissue_masks)
            if self.USE_ANNOTATION2:
                self._annotations2 = np.array(self._annotations2)

    @staticmethod
    def _load_graph(i, row):
        graph = load_graphs(str(row["graph_path"]))[0][0]
        graph.readonly()
        return graph

    def _downsample(self, array):
        if self.downsample != 1:
            new_size = (
                array.shape[0] // self.downsample,
                array.shape[1] // self.downsample,
            )
            array = cv2.resize(
                array,
                new_size,
                interpolation=cv2.INTER_NEAREST,
            )
        return array

    def _load_h5(self, path):
        try:
            with h5py.File(path, "r") as file:
                if "default_key_0" in file:
                    content = file["default_key_0"][()]
                elif "default_key" in file:
                    content = file["default_key"][()]
                else:
                    raise NotImplementedError(
                        f"Superpixels not found in keys. Available are: {file.keys()}"
                    )
        except OSError as e:
            print(f"Could not open {path}")
            raise e
        return self._downsample(content)

    def _load_image(self, path):
        image = read_image(path)
        return self._downsample(image)

    def _load_datapoint(self, i, row):
        super()._load_datapoint(i, row)
        self._graphs.append(self._load_graph(i, row))
        if self.return_segmentation_info:
            self._superpixels.append(self._load_h5(row["superpixel_path"]))
            if "annotation_path" in row:
                self._annotations.append(self._load_image(row["annotation_path"]))
            self._tissue_masks.append(self._load_image(row["tissue_mask_path"]))
            if self.USE_ANNOTATION2 and "annotation2_path" in row:
                self._annotations2.append(self._load_image(row["annotation2_path"]))

    def _load_image_level_labels(self, label_mapper):
        labels = list()
        for name in self._names:
            labels.append((label_mapper[name]))
        return torch.as_tensor(labels)

    @property
    def graphs(self):
        return self._graphs

    @property
    def graph_labels(self):
        return self._graph_labels

    @property
    def superpixels(self):
        return self._superpixels

    @property
    def tissue_masks(self):
        return self._tissue_masks

    @property
    def annotations(self):
        return self._annotations

    @property
    def annotations2(self):
        return self._annotations2

    @property
    def indices(self):
        if self.mode == "tissue":
            return self._tissue_indices
        elif self.mode == "image":
            return self._image_indices
        else:
            raise NotImplementedError

    def _select_graph_features(self, centroid_features):
        for graph, image_size in zip(self.graphs, self.image_sizes):
            assert (
                len(graph.ndata[FEATURES].shape) == 2
            ), f"Cannot use GraphClassificationDataset when the preprocessing was run with augmentations"
            if centroid_features == "only":
                features = (graph.ndata[CENTROID] / torch.Tensor(image_size)).to(
                    torch.float32
                )
                graph.ndata.pop(FEATURES)
            else:
                features = graph.ndata.pop(FEATURES).to(torch.float32)
                if self.mean is not None and self.std is not None:
                    features = (features - self.mean) / self.std

                if centroid_features == "cat":
                    features = torch.cat(
                        [
                            features,
                            (graph.ndata[CENTROID] / torch.Tensor(image_size)).to(
                                torch.float32
                            ),
                        ],
                        dim=1,
                    )
            graph.ndata[GNN_NODE_FEAT_IN] = features

    def _compute_graph_labels(self):
        graph_labels = list()
        for graph in self.graphs:
            node_labels = graph.ndata[LABEL]
            graph_label = self._to_onehot_with_ignore(pd.unique(node_labels.numpy()))
            graph_label = graph_label.sum(axis=0)
            graph_labels.append(graph_label)
        return graph_labels

    @staticmethod
    def _get_indices_in_bounding_box(
        centroids: torch.Tensor, bounding_box: Tuple[int, int, int, int]
    ) -> torch.Tensor:
        """Returns the node indices of the nodes contained in the bounding box

        Args:
            centroids (torch.Tensor): Tensor of centroid locations. Shape: nr_superpixels x 2
            bounding_box (Tuple[int, int, int, int]): Bounding box to consider (x_min, y_min, x_max, y_max)

        Returns:
            torch.Tensor: Tensor of indices that lie within the bounding box
        """
        x_min, y_min, x_max, y_max = bounding_box
        node_mask = (
            (centroids[:, 0] >= x_min)
            & (centroids[:, 1] >= y_min)
            & (centroids[:, 0] <= x_max)
            & (centroids[:, 1] <= y_max)
        )
        node_indices = torch.where(node_mask)[0]
        return node_indices

    @staticmethod
    def _generate_subgraph(
        graph: dgl.DGLGraph, node_indices: torch.Tensor
    ) -> dgl.DGLGraph:
        """Generates a subgraph with only the nodes and edges in node_indices

        Args:
            graph (dgl.DGLGraph): Input graph
            node_indices (torch.Tensor): Node indices to consider

        Returns:
            dgl.DGLGraph: A subgraph with the subset of nodes and edges
        """
        subgraph = graph.subgraph(node_indices)
        for key, item in graph.ndata.items():
            subgraph.ndata[key] = item[subgraph.ndata["_ID"]]
        for key, item in graph.edata.items():
            subgraph.edata[key] = item[subgraph.edata["_ID"]]
        return subgraph

    @staticmethod
    def _get_random_patch(
        full_size: Tuple[int, int], patch_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Get a random bounding box from a given image size and desired patch size

        Args:
            full_size (Tuple[int, int]): Size of the full image
            patch_size (Tuple[int, int]): Desired size of the patch

        Returns:
            Tuple[int, int, int, int]: A bounding box of the patch
        """
        x_min = np.random.randint(full_size[0] - patch_size[0])
        y_min = np.random.randint(full_size[1] - patch_size[1])
        return (x_min, y_min, x_min + patch_size[0], y_min + patch_size[1])

    def _build_datapoint(self, graph, node_labels, index):
        if self.mode == "tissue":
            return GraphDatapoint(
                graph=graph,
                graph_label=self.graph_labels[index],
                node_labels=node_labels,
                name=self.names[index],
                instance_map=self.superpixels[index]
                if self.return_segmentation_info
                else None,
                segmentation_mask=self.annotations[index]
                if self.return_segmentation_info
                else None,
                tissue_mask=self.tissue_masks[index]
                if self.return_segmentation_info
                else None,
                additional_segmentation_mask=self.annotations2[index]
                if self.USE_ANNOTATION2
                else None,
            )
        else:
            assert node_labels is None
            return GraphDatapoint(
                graph=graph,
                graph_label=self.graph_labels[index],
                node_labels=None,
                name=self.names[index],
                instance_map=self.superpixels[index]
                if self.return_segmentation_info
                else None,
                tissue_mask=self.tissue_masks[index]
                if self.return_segmentation_info
                else None,
            )

    def _get_random_subgraph(self, graph, node_labels, image_size):
        bounding_box = self._get_random_patch(
            full_size=image_size, patch_size=self.patch_size
        )
        relevant_nodes = self._get_indices_in_bounding_box(
            graph.ndata[CENTROID], bounding_box
        )
        graph = self._generate_subgraph(graph, relevant_nodes)
        if node_labels is not None:
            node_labels = node_labels[relevant_nodes]
        return graph, node_labels

    def set_mode(self, mode):
        valid_modes = ["image", "tissue"]
        assert (
            mode in valid_modes
        ), f"Dataset mode must be from {valid_modes}, but is {mode}"
        self.mode = mode

    def __getitem__(self, index: int) -> GraphDatapoint:
        """Returns a sample (patch) of graph i

        Args:
            index (int): Index of graph

        Returns:
            GraphDatapoint: Subgraph and graph label and the node labels
        """
        if isinstance(index, str):
            index = self.name_to_index[index]
        assert (
            index in self.indices
        ), f"Index ({index}) not in range of datapoints ({self.indices}) for this mode ({self.mode})."
        graph = self.graphs[index]
        if self.mode == "tissue":
            node_labels = graph.ndata[LABEL]
        else:
            node_labels = None

        # Random patch sampling
        if self.patch_size is not None:
            image_size = self.image_sizes[index]
            graph, node_labels = self._get_random_subgraph(
                graph, node_labels, image_size
            )

        return self._build_datapoint(graph, node_labels, index)

    def __len__(self) -> int:
        """Number of graphs in the dataset

        Returns:
            int: Length of the dataset
        """
        if self.mode == "tissue":
            return len(self._tissue_indices)
        elif self.mode == "image":
            assert len(self.graphs) == len(self._image_indices)
            return len(self.graphs)


class AugmentedGraphClassificationDataset(GraphClassificationDataset):
    """Dataset variation used for extracted and dumped graphs with augmentations"""

    def __init__(self, augmentation_mode: Optional[str] = None, **kwargs) -> None:
        self.augmentation_mode = augmentation_mode
        super().__init__(**kwargs)

    def set_augmentation_mode(self, augmentation_mode: Optional[str] = None) -> None:
        """Set a fixed augmentation to be used

        Args:
            augmentation (Optional[int], optional): Augmentation index to use. None refers to randomly sample features for each node. Defaults to None.
        """
        self.augmentation_mode = augmentation_mode

    def _select_graph_features(self, centroid_features: str) -> None:
        """Skips the feature selection step as it is done during data loading

        Args:
            centroid_features (str): Centroid features to select during data loading
        """
        self.centroid_features = centroid_features

    def __getitem__(self, index: int) -> Any:
        """Get a graph to train with. Randomly samples features from the available features.

        Args:
            index (int): Index of the graph

        Returns:
            Any: Return tuple depending on the arguments
        """
        if isinstance(index, str):
            index = self.name_to_index[index]
        assert (
            index in self.indices
        ), f"Index ({index}) not in range of datapoints ({self.indices}) for this mode ({self.mode})."
        image_size = self.image_sizes[index]
        graph = self.graphs[index]

        augmented_graph = dgl.DGLGraph(graph_data=graph)
        augmented_graph.ndata[CENTROID] = graph.ndata[CENTROID]
        if self.mode == "tissue":
            augmented_graph.ndata[LABEL] = graph.ndata[LABEL]

        # Get features
        if self.centroid_features == "only":
            features = (augmented_graph.ndata[CENTROID] / torch.Tensor(image_size)).to(
                torch.float32
            )
        else:
            assert (
                len(graph.ndata[FEATURES].shape) == 3
            ), f"Cannot use AugmentedDataset when the preprocessing was not run with augmentations"
            nr_nodes, nr_augmentations, _ = graph.ndata[FEATURES].shape

            # Sample based on augmentation mode
            if self.augmentation_mode == "graph":
                sample = torch.ones(size=(nr_nodes,), dtype=torch.long) * torch.randint(
                    low=0, high=nr_augmentations, size=(1,)
                )
            elif self.augmentation_mode == "node":
                sample = torch.randint(
                    low=0, high=nr_augmentations, size=(nr_nodes,), dtype=torch.long
                )
            else:
                sample = torch.zeros(size=(nr_nodes,), dtype=torch.long)

            # Select features to use
            features = graph.ndata[FEATURES][torch.arange(nr_nodes), sample].to(
                torch.float32
            )

            if self.mean is not None and self.std is not None:
                features = (features - self.mean) / self.std

            if self.centroid_features == "cat":
                features = torch.cat(
                    [
                        features,
                        (augmented_graph.ndata[CENTROID] / torch.Tensor(image_size)).to(
                            torch.float32
                        ),
                    ],
                    dim=1,
                )
        augmented_graph.ndata[GNN_NODE_FEAT_IN] = features
        if self.mode == "tissue":
            node_labels = augmented_graph.ndata[LABEL]
        else:
            node_labels = None

        # Random patch sampling
        if self.patch_size is not None:
            image_size = self.image_sizes[index]
            augmented_graph, node_labels = self._get_random_subgraph(
                augmented_graph, node_labels, image_size
            )

        return self._build_datapoint(augmented_graph, node_labels, index)


class ImageDataset(BaseDataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        downsample_factor: int,
        num_classes: int = 4,
        background_index: int = 4,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        assert (
            "processed_image_path" in metadata
        ), f"Metadata lacks processed image path ({metadata.columns})"
        assert (
            "annotation_path" in metadata
        ), f"Metadata lacks annotation path ({metadata.columns})"
        super().__init__(metadata, num_classes, background_index, **kwargs)
        self.downsample_factor = downsample_factor
        self.images, self.annotations = self._load_images()
        self.tissue_masks = self._load_tissue_masks()
        self.normalizer = Normalize(mean, std)
        self.additional_annotation = "annotation2_path" in metadata
        if self.additional_annotation:
            self.annotations2 = self._load_annotation2()

    def _load_tissue_masks(self):
        mask_paths = self.metadata["tissue_mask_path"].tolist()
        tissue_masks = list()
        for mask_path in mask_paths:
            tissue_mask = read_image(mask_path)
            if self.downsample_factor != 1:
                new_size = (
                    math.floor(tissue_mask.shape[0] / self.downsample_factor),
                    math.floor(tissue_mask.shape[1] / self.downsample_factor),
                )
                tissue_mask = cv2.resize(
                    tissue_mask, new_size, interpolation=cv2.INTER_NEAREST
                )
            tissue_masks.append(tissue_mask)
        return torch.from_numpy(np.array(tissue_masks))

    def _load_annotation2(self):
        assert "annotation2_path" in self.metadata
        annotation_paths = self.metadata["annotation2_path"].tolist()
        annotations = list()
        for annotation_path in annotation_paths:
            annotation = read_image(annotation_path)
            if self.downsample_factor != 1:
                new_size = (
                    math.floor(annotation.shape[0] / self.downsample_factor),
                    math.floor(annotation.shape[1] / self.downsample_factor),
                )
                annotation = cv2.resize(
                    annotation, new_size, interpolation=cv2.INTER_NEAREST
                )
            annotations.append(annotation)
        return torch.from_numpy(np.array(annotations))

    def _load_images(self):
        image_paths = self.metadata["processed_image_path"].tolist()
        annotation_paths = self.metadata["annotation_path"].tolist()
        images = list()
        annotations = list()
        for image_path, annotation_path in tqdm(
            zip(image_paths, annotation_paths),
            total=len(image_paths),
            desc="dataset_loading",
        ):
            image = read_image(image_path)
            annotation = read_image(annotation_path)
            if self.downsample_factor != 1:
                new_size = (
                    math.floor(image.shape[0] / self.downsample_factor),
                    math.floor(image.shape[1] / self.downsample_factor),
                )
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
                annotation = cv2.resize(
                    annotation, new_size, interpolation=cv2.INTER_NEAREST
                )
            images.append(image)
            annotations.append(annotation)
        return torch.from_numpy(np.array(images)), torch.from_numpy(
            np.array(annotations)
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index].permute(2, 0, 1).to(torch.float32) / 255
        normalized_image = self.normalizer(image)

        return ImageDatapoint(
            image=normalized_image,
            segmentation_mask=self.annotations[index],
            tissue_mask=self.tissue_masks[index],
            name=self.names[index],
            additional_segmentation_mask=self.annotations2[index]
            if self.additional_annotation
            else None,
        )

    def __len__(self) -> int:
        return len(self.images)


class PatchClassificationDataset(ImageDataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        patch_size: int,
        stride: int,
        downsample_factor: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        augmentations: Optional[List[Dict]] = None,
        num_classes: int = 4,
        background_index: int = 4,
        class_threshold: float = 0.0001,
        **kwargs,
    ) -> None:
        assert (
            "processed_image_path" in metadata
        ), f"Metadata lacks processed image path ({metadata.columns})"
        assert (
            "annotation_path" in metadata
        ), f"Metadata lacks annotation path ({metadata.columns})"
        super().__init__(
            metadata=metadata,
            downsample_factor=downsample_factor,
            num_classes=num_classes,
            background_index=background_index,
            **kwargs,
        )
        self.patch_size = patch_size
        self.stride = stride
        self.class_threshold = class_threshold
        self.patches = self._generate_patches()
        self.labels = self._generate_labels()
        self.masks = self._generate_tissue_masks()
        self.augmentor = self._get_augmentor(augmentations, mean, std)

    def _generate_patches(self):
        patches = self.images.unfold(1, self.patch_size, self.stride).unfold(
            2, self.patch_size, self.stride
        )
        patches = patches.reshape([-1, 3, self.patch_size, self.patch_size])
        return patches

    def _to_unique_onehot(self, annotation):
        counts = fast_histogram(annotation, self.num_classes + 1)
        counts = counts / counts.sum()
        unique_annotation = np.where(counts > self.class_threshold)[0]
        return self._to_onehot_with_ignore(unique_annotation).sum(axis=0).numpy()

    def _generate_labels(self):
        labels = self.annotations.unfold(1, self.patch_size, self.stride).unfold(
            2, self.patch_size, self.stride
        )
        self.patch_annotations = labels.reshape([-1, self.patch_size, self.patch_size])
        return torch.as_tensor(
            np.array(
                list(map(self._to_unique_onehot, self.patch_annotations)),
                dtype=np.uint8,
            )
        )

    def _generate_tissue_masks(self):
        masks = self.tissue_masks.unfold(1, self.patch_size, self.stride).unfold(
            2, self.patch_size, self.stride
        )
        masks = masks.reshape([-1, self.patch_size, self.patch_size])
        masks = masks // 255
        return masks

    @staticmethod
    def _get_augmentor(
        augmentations: Optional[List[Dict]], mean: torch.Tensor, std: torch.Tensor
    ):
        augmentation_pipeline = list()
        if augmentations is not None:
            augmentation_pipeline.append(ToPILImage())
            if "rotation" in augmentations:
                augmentation_pipeline.append(
                    RandomRotation(degrees=augmentations["rotation"]["degrees"])
                )
                if "crop" in augmentations["rotation"]:
                    augmentation_pipeline.append(
                        CenterCrop(augmentations["rotation"]["crop"])
                    )
            if "flip" in augmentations:
                augmentation_pipeline.append(
                    Compose([RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5)])
                )
            if "color_jitter" in augmentations:
                augmentation_pipeline.append(
                    ColorJitter(**augmentations["color_jitter"])
                )
            augmentation_pipeline.append(ToTensor())
        augmentation_pipeline.append(Normalize(mean, std))
        return Compose(augmentation_pipeline)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[index].to(torch.float32) / 255.0
        label = self.labels[index]
        augmented_patch = self.augmentor(patch)
        return augmented_patch, label

    def __len__(self) -> int:
        return len(self.patches)

    def drop_tissueless_patches(self, minimum_fraction: 0.0) -> None:
        """Drops patches that have less than minimum_fraction of tissue on them

        Args:
            minimum_fraction (0.0): Minimum fraction of tissues
        """
        tissue_percentages = self.masks.sum(dim=1).sum(dim=1) / float(
            self.patch_size * self.patch_size
        )
        patches_with_tissue = tissue_percentages > minimum_fraction
        self.patches = self.patches[patches_with_tissue]
        self.labels = self.labels[patches_with_tissue]
        self.masks = self.masks[patches_with_tissue]
        self.patch_annotations = self.patch_annotations[patches_with_tissue]

    def drop_unlablled_patches(self) -> None:
        patches_with_at_least_one_class = self.labels.sum(dim=1) != 0
        self.patches = self.patches[patches_with_at_least_one_class]
        self.labels = self.labels[patches_with_at_least_one_class]
        self.masks = self.masks[patches_with_at_least_one_class]
        self.patch_annotations = self.patch_annotations[patches_with_at_least_one_class]

    def drop_confusing_patches(self) -> None:
        """Drops patches that contain more than one class"""
        patches_with_single_class = self.labels.sum(dim=1) == 1
        self.patches = self.patches[patches_with_single_class]
        self.labels = self.labels[patches_with_single_class]
        self.masks = self.masks[patches_with_single_class]
        self.patch_annotations = self.patch_annotations[patches_with_single_class]

    def get_class_weights(self) -> torch.Tensor:
        _, classes, counts = self.labels.unique(
            dim=0, return_inverse=True, return_counts=True
        )
        frequencies = 1 / counts.to(torch.float32)
        return frequencies[classes]
