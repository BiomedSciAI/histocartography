"""This module handles all the graph building"""

import logging
from abc import abstractmethod

import cv2
import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs, save_graphs

from constants import CENTROID, GNN_EDGE_FEAT, GNN_NODE_FEAT_IN
from utils import PipelineStep


class BaseGraphBuilder(PipelineStep):
    """
    Base interface class for graph building.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, structure: np.ndarray, features: torch.Tensor) -> dgl.DGLGraph:
        """Generates a graph with a given structure and features

        Args:
            structure (np.array): Structure, depending on the graph can be superpixel connectivity, or centroids
            features (torch.Tensor): Features of each node. Shape (nr_nodes, nr_features)

        Returns:
            dgl.DGLGraph: The constructed graph
        """

        # add nodes
        num_nodes = features.shape[0]
        graph = dgl.DGLGraph()
        graph.add_nodes(num_nodes)

        # add node features
        self._set_node_features(features, graph)

        # build edges
        self._build_topology(
            structure,
            graph,
        )
        return graph

    def process_and_save(
        self, structure: np.ndarray, features: torch.Tensor, output_name: str
    ) -> dgl.DGLGraph:
        assert (
            self.base_path is not None
        ), f"Can only save intermediate output if base_path was not None when constructing the object"
        output_path = self.output_dir / f"{output_name}.bin"
        if output_path.exists():
            logging.info(
                f"Output of {output_name} already exists, using it instead of recomputing"
            )
            graphs, _ = load_graphs(str(output_path))
            assert len(graphs) == 1
            graph = graphs[0]
        else:
            graph = self.process(structure=structure, features=features)
            save_graphs(str(output_path), [graph])
        return graph

    @staticmethod
    def _set_node_features(features: torch.Tensor, graph: dgl.DGLGraph) -> None:
        """Set the provided node features

        Args:
            features (torch.Tensor): Node features
            graph (dgl.DGLGraph): Graph to add the features to
        """
        graph.ndata[GNN_NODE_FEAT_IN] = features

    @abstractmethod
    def _build_topology(self, instances: np.ndarray, graph: dgl.DGLGraph) -> None:
        """Generate the graph topology from the provided structure

        Args:
            instances (np.array): Graph structure
            graph (dgl.DGLGraph): Graph to add the edges
        """

    def __repr__(self) -> str:
        """Representation of a graph builder

        Returns:
            str: Representation of a graph builder
        """
        return f'{self.__class__.__name__}({",".join([f"{k}={v}" for k, v in vars(self).items()])})'


class RAGGraphBuilder(BaseGraphBuilder):
    """
    Super-pixel Graphs class for graph building.
    """

    def __init__(self, kernel_size: int = 5, **kwargs) -> None:
        """Create a graph builder that uses a provided kernel size to detect connectivity

        Args:
            kernel_size (int, optional): Size of the kernel to detect connectivity. Defaults to 5.
        """
        super(RAGGraphBuilder, self).__init__()
        logging.debug("*** RAG Graph Builder ***")
        self.kernel_size = kernel_size
        super().__init__(**kwargs)

    def _build_topology(self, instances: np.ndarray, graph: dgl.DGLGraph) -> None:
        """Create the graph topology from the connectivty of the provided superpixels

        Args:
            instances (np.array): Superpixels
            graph (dgl.DGLGraph): Graph to add the edges to
        """
        instance_ids = np.sort(pd.unique(np.ravel(instances))).astype(int)
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        adjacency = np.zeros(shape=(len(instance_ids), len(instance_ids)))
        for instance_id in instance_ids:
            mask = np.array(instances == instance_id, np.uint8)
            dilation = cv2.dilate(mask, kernel, iterations=1)
            boundary = dilation - mask
            mask = boundary * instances
            idx = np.array(pd.unique(np.ravel(mask))).astype(int)[1:]  # remove 0
            instance_id -= 1  # because instance_map id starts from 1
            idx -= 1  # because instance_map id starts from 1
            adjacency[instance_id, idx] = 1

        edge_list = np.nonzero(adjacency)
        graph.add_edges(list(edge_list[0]), list(edge_list[1]))
