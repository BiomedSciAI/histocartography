"""Dataloader for precomputed graphs in .bin format"""
from sys import meta_path
from typing import List, Tuple

import h5py
import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs
from dgl.graph import DGLGraph
from PIL import Image
from torch.utils.data import Dataset

from constants import CENTROID, LABEL


class GraphClassificationDataset(Dataset):
    def __init__(
        self, metadata: pd.DataFrame, patch_size: Tuple[int, int], nr_classes: int = 4, ignore_class: Tuple[int, None] = 4,
    ) -> None:
        self._check_metadata(metadata)
        self.metadata = metadata
        self.names, self.graphs = self._load_graphs(self.metadata)
        self.image_sizes = self._load_image_sizes(self.metadata)
        self.patch_size = patch_size
        self.nr_classes = nr_classes
        self.ignore_class = ignore_class

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

    def __getitem__(self, index: int) -> Tuple[dgl.DGLGraph, int]:
        """Returns a sample (patch) of graph i

        Args:
            index (int): Index of graph

        Returns:
            Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]: Subgraph and corresponding one-hot encoded graph label and the (not-onehot) node label
        """
        graph = self.graphs[index]
        image_size = self.image_sizes[index]
        annotation = graph.ndata[LABEL]

        # Random patch sampling
        if self.patch_size is not None:
            bounding_box = self._get_random_patch(
                full_size=image_size, patch_size=self.patch_size
            )
            relevant_nodes = self._get_indices_in_bounding_box(
                graph.ndata[CENTROID], bounding_box
            )
            graph = self._generate_subgraph(graph, relevant_nodes)
            annotation = annotation[relevant_nodes]

        # Label extraction
        graph_label = pd.unique(annotation.numpy())
        if self.ignore_class is not None:
            graph_label = graph_label[graph_label != self.ignore_class]
        one_hot_graph_label = torch.zeros(self.nr_classes, dtype=torch.int)
        one_hot_graph_label[graph_label.astype(int)] = 1
        return graph, one_hot_graph_label, annotation

    def __len__(self) -> int:
        """Number of graphs in the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.metadata)
