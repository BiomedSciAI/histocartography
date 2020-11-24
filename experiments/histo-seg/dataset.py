"""Dataloader for precomputed graphs in .bin format"""
from typing import Dict, List, Optional, Tuple, Union

import cv2
import dgl
import h5py
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs
from dgl.graph import DGLGraph
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
from typing_extensions import final

from constants import CENTROID, FEATURES, GNN_NODE_FEAT_IN, LABEL
from utils import read_image


class BaseDataset(Dataset):
    def __init__(self, metadata, patch_size, num_classes, background_index) -> None:
        self._check_metadata(metadata)
        self.metadata = metadata
        self.image_sizes = self._load_image_sizes(metadata)
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.background_index = background_index

    @staticmethod
    def _check_metadata(metadata: pd.DataFrame) -> None:
        """Checks that the given metadata has a valid format and all referenced files exist

        Args:
            metadata (pd.DataFrame): Metadata dataframe
        """
        assert not metadata.isna().any().any(), f"Some entries in metadata are NaN"
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

    @staticmethod
    def _load_image_sizes(metadata: pd.DataFrame) -> List[Tuple[int, int]]:
        image_sizes = list()
        for _, row in metadata.iterrows():
            image_sizes.append((row.height, row.width))
        return image_sizes

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


class GraphClassificationDataset(BaseDataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        patch_size: Optional[Tuple[int, int]],
        num_classes: int = 4,
        background_index: int = 4,
        centroid_features: str = "no",
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        return_segmentation_info: bool = False,
        segmentation_downsample_ratio: int = 1,
    ) -> None:
        assert centroid_features in [
            "no",
            "cat",
            "only",
        ], f"centroid_features must be in [no, cat, only] but is {centroid_features}"
        assert (
            "graph_path" in metadata
        ), f"Metadata lacks graph path ({metadata.columns})"
        super().__init__(metadata, patch_size, num_classes, background_index)
        self.mean = mean
        self.std = std
        self.names, self.graphs = self._load_graphs(metadata)
        self.name_to_index = dict(zip(self.names, range(len(self.names))))
        self.graph_labels = self._compute_graph_labels()
        self._select_graph_features(centroid_features)
        self.return_segmentation_info = return_segmentation_info
        self.downsample = segmentation_downsample_ratio
        if return_segmentation_info:
            self.superpixels = list()
            self.annotations = list()
            for i, row in self.metadata.iterrows():
                annotation_path = row["annotation_path"]
                superpixel_path = row["superpixel_path"]
                annotation = read_image(annotation_path)
                with h5py.File(superpixel_path, "r") as file:
                    superpixel = file["default_key"][()]
                if self.downsample != 1:
                    new_size = (
                        annotation.shape[0] // self.downsample,
                        annotation.shape[1] // self.downsample,
                    )
                    annotation = cv2.resize(
                        annotation,
                        new_size,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    superpixel = cv2.resize(
                        superpixel,
                        new_size,
                        interpolation=cv2.INTER_NEAREST,
                    )
                self.superpixels.append(superpixel)
                self.annotations.append(annotation)

    @staticmethod
    def _load_graphs(metadata: pd.DataFrame) -> Tuple[List[str], List[DGLGraph]]:
        """Loads all graphs from disk into memory

        Args:
            metadata (pd.DataFrame): Metadata dataframe

        Returns:
            Tuple[List[str], List[DGLGraph]]: A list of names and a list of graphs
        """
        names, graphs = list(), list()
        for name, row in metadata.iterrows():
            names.append(name)
            graphs.append(load_graphs(str(row["graph_path"]))[0][0])
        return names, graphs

    def _select_graph_features(self, centroid_features):
        for graph, image_size in zip(self.graphs, self.image_sizes):
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

    def _compute_graph_labels(self):
        graph_labels = list()
        for graph in self.graphs:
            node_labels = graph.ndata[LABEL]
            graph_label = self._to_onehot_with_ignore(pd.unique(node_labels.numpy()))
            graph_label = graph_label.sum(axis=0)
            graph_labels.append(graph_label)
        return graph_labels

    def __getitem__(
        self, index: int
    ) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
        """Returns a sample (patch) of graph i

        Args:
            index (int): Index of graph

        Returns:
            Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]: Subgraph and graph label and the node labels
        """
        if isinstance(index, str):
            index = self.name_to_index[index]
        graph = self.graphs[index]
        image_size = self.image_sizes[index]
        node_labels = graph.ndata[LABEL]
        graph_label = self.graph_labels[index]

        # Random patch sampling
        if self.patch_size is not None:
            bounding_box = self._get_random_patch(
                full_size=image_size, patch_size=self.patch_size
            )
            relevant_nodes = self._get_indices_in_bounding_box(
                graph.ndata[CENTROID], bounding_box
            )
            graph = self._generate_subgraph(graph, relevant_nodes)
            node_labels = node_labels[relevant_nodes]

        if self.return_segmentation_info:
            superpixel = self.superpixels[index]
            annotation = self.annotations[index]
            return graph, graph_label, node_labels, annotation, superpixel
        return graph, graph_label, node_labels

    def __len__(self) -> int:
        """Number of graphs in the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.graphs)


class PatchClassificationDataset(BaseDataset):
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
    ) -> None:
        assert (
            "processed_image_path" in metadata
        ), f"Metadata lacks processed image path ({metadata.columns})"
        assert (
            "annotation_path" in metadata
        ), f"Metadata lacks annotation path ({metadata.columns})"
        super().__init__(metadata, patch_size, num_classes, background_index)
        self.downsample_factor = downsample_factor
        self.stride = stride
        self.images, self.annotations = self._load_images()
        self.patches = self._generate_patches()
        self.labels = self._generate_labels()
        self.augmentor = self._get_augmentor(augmentations, mean, std)

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
                    image.shape[0] // self.downsample_factor,
                    image.shape[1] // self.downsample_factor,
                )
                image = cv2.resize(image, new_size)
                annotation = cv2.resize(annotation, new_size)
            images.append(image)
            annotations.append(annotation)
        return torch.from_numpy(np.array(images)), torch.from_numpy(
            np.array(annotations)
        )

    def _generate_patches(self):
        patches = self.images.unfold(1, self.patch_size, self.stride).unfold(
            2, self.patch_size, self.stride
        )
        patches = patches.reshape([-1, 3, self.patch_size, self.patch_size])
        return patches

    def _to_unique_onehot(self, annotation):
        unique_annotation = pd.unique(annotation.numpy().ravel())
        return self._to_onehot_with_ignore(unique_annotation).sum(axis=0).numpy()

    def _generate_labels(self):
        labels = self.annotations.unfold(1, self.patch_size, self.stride).unfold(
            2, self.patch_size, self.stride
        )
        labels = labels.reshape([-1, self.patch_size, self.patch_size])
        return torch.as_tensor(
            np.array(list(map(self._to_unique_onehot, labels)), dtype=np.uint8)
        )

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


def collate(
    samples: List[Tuple[DGLGraph, torch.Tensor, torch.Tensor]]
) -> Tuple[DGLGraph, torch.Tensor, torch.Tensor]:
    """Aggregate a batch by performing the following:
       Create a graph with disconnected components using dgl.batch
       Stack the graph labels one-hot encoded labels (to shape B x nr_classes)
       Concatenate the node labels to a single vector (graph association can be read from graph.batch_num_nodes)

    Args:
        samples (List[Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]]): List of unaggregated samples

    Returns:
        Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]: Aggregated graph and labels
    """
    graphs, graph_labels, node_labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.stack(graph_labels), torch.cat(node_labels)


def collate_valid(
    samples: List[Tuple[DGLGraph, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]]
) -> Tuple[DGLGraph, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Aggregate a batch by performing the following:
       Create a graph with disconnected components using dgl.batch
       Stack the graph labels one-hot encoded labels (to shape B x nr_classes)
       Concatenate the node labels to a single vector (graph association can be read from graph.batch_num_nodes)

    Args:
        samples (List[Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]]): List of unaggregated samples

    Returns:
        Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]: Aggregated graph and labels
    """
    graphs, graph_labels, node_labels, annotations, superpixels = map(
        list, zip(*samples)
    )
    return (
        dgl.batch(graphs),
        torch.stack(graph_labels),
        torch.cat(node_labels),
        np.stack(annotations),
        np.stack(superpixels),
    )
