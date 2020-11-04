"""Dataloader for precomputed graphs in .bin format"""
from typing import List, Tuple, Union

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs
from dgl.graph import DGLGraph
from torch.utils.data import Dataset

from constants import CENTROID, LABEL


class GraphClassificationDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        patch_size: Union[None, Tuple[int, int]],
        num_classes: int = 4,
        background_index: int = 4,
    ) -> None:
        self._check_metadata(metadata)
        self.names, self.graphs = self._load_graphs(metadata)
        self.image_sizes = self._load_image_sizes(metadata)
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.background_index = background_index
        self.name_to_index = dict(zip(self.names, range(len(self.names))))

    @staticmethod
    def _check_metadata(metadata: pd.DataFrame) -> None:
        """Checks that the given metadata has a valid format and all referenced files exist

        Args:
            metadata (pd.DataFrame): Metadata dataframe
        """
        assert not metadata.isna().any().any(), f"Some entries in metadata are NaN"
        assert (
            "graph_path" in metadata
        ), f"Metadata lacks graph path ({metadata.columns})"
        for name, row in metadata.iterrows():
            assert (
                row.graph_path.exists()
            ), f"Graph {name} referenced in metadata does not exist: {row.graph_path}"

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

    @staticmethod
    def _load_image_sizes(metadata: pd.DataFrame) -> List[Tuple[int, int]]:
        image_sizes = list()
        for _, row in metadata.iterrows():
            image_sizes.append((row.height, row.width))
        return image_sizes

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

    def _to_onehot_with_ignore(self, input_vector: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
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

    def __getitem__(self, index: int) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
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

        # Label extraction
        graph_label = self._to_onehot_with_ignore(pd.unique(node_labels.numpy()))
        graph_label = graph_label.sum(axis=0)

        return graph, graph_label, node_labels

    def __len__(self) -> int:
        """Number of graphs in the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.graphs)
