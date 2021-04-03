"""This module handles all the graph building"""

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import cv2
import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs, save_graphs
from skimage.measure import regionprops
from sklearn.neighbors import kneighbors_graph

from .constants import CENTROID, FEATURES, LABEL
from ..pipeline import PipelineStep
from .utils import fast_histogram


def two_hop_neighborhood(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    """Increases the connectivity of a given graph by an additional hop

    Args:
        graph (dgl.DGLGraph): Input graph

    Returns:
        dgl.DGLGraph: Output graph
    """
    A = graph.adjacency_matrix().to_dense()
    A_tilde = (1.0 * ((A + A.matmul(A)) >= 1)) - torch.eye(A.shape[0])
    ngraph = nx.convert_matrix.from_numpy_matrix(A_tilde.numpy())
    new_graph = dgl.DGLGraph()
    new_graph.from_networkx(ngraph)
    for k, v in graph.ndata.items():
        new_graph.ndata[k] = v
    for k, v in graph.edata.items():
        new_graph.edata[k] = v
    return new_graph


class BaseGraphBuilder(PipelineStep):
    """
    Base interface class for graph building.
    """

    def __init__(
        self, nr_classes: int = 6, background_class: int = 4, add_loc_feats: bool = False, **kwargs: Any
    ) -> None:
        """
        Base Graph Builder constructor.

        Args:
            nr_classes (int): Number of node labels. Used only if setting node labels.
            background_class (int): Number of node labels. Used only if setting node labels.
            add_loc_feats (bool): If we should add location-based features (ie the centroids) to the node features.
                                  Defaults to False.
        """
        self.add_loc_feats = add_loc_feats
        super().__init__(**kwargs)
        self.nr_classes = nr_classes
        self.background_class = background_class

    def _process(  # type: ignore[override]
        self,
        structure: np.ndarray,
        features: torch.Tensor,
        image_size: Tuple[int, int] = None,
        annotation: Optional[np.ndarray] = None,
    ) -> dgl.DGLGraph:
        """Generates a graph with a given structure and features

        Args:
            structure (np.array): Structure, depending on the graph can be superpixel
                                  connectivity, or centroids
            features (torch.Tensor): Features of each node. Shape (nr_nodes, nr_features)
            image_size (Tuple[int, int]): Image size corresponding to the graph.
                                          Defaults to None.
            annotation (Union[None, np.array], optional): Optional node level to include.
                                                          Defaults to None.

        Returns:
            dgl.DGLGraph: The constructed graph
        """
        # add nodes
        num_nodes = features.shape[0]
        graph = dgl.DGLGraph()
        graph.add_nodes(num_nodes)

        # add image size as graph data
        if image_size is not None:
            graph.gdata = {"image_size": image_size}

        # add node features
        self._set_node_centroids(structure, graph)
        self._set_node_features(features, graph)
        if annotation is not None:
            self._set_node_labels(structure, annotation, graph)

        # build edges
        self._build_topology(
            structure,
            graph,
        )
        return graph

    def _process_and_save(  # type: ignore[override]
        self,
        structure: np.ndarray,
        features: torch.Tensor,
        annotation: Optional[np.ndarray] = None,
        *,
        output_name: str,
    ) -> dgl.DGLGraph:
        """Process and save in provided directory

        Args:
            output_name (str): Name of output file
            structure (np.ndarray): Structure, dependeing on the graph can be superpixel
                                    connectivity or centroids
            features (torch.Tensor): Features of each node. Shape (nr_nodes, nr_features)
            annotation (Optional[np.ndarray], optional): Optional node level to include.
                                                         Defaults to None.

        Returns:
            dgl.DGLGraph: [description]
        """
        assert (
            self.save_path is not None
        ), "Can only save intermediate output if base_path was not None during construction"
        output_path = self.output_dir / f"{output_name}.bin"
        if output_path.exists():
            logging.info(
                f"Output of {output_name} already exists, using it instead of recomputing"
            )
            graphs, _ = load_graphs(str(output_path))
            assert len(graphs) == 1
            graph = graphs[0]
        else:
            graph = self._process(
                structure=structure, features=features, annotation=annotation
            )
            save_graphs(str(output_path), [graph])
        return graph

    def _set_node_features(self, features: torch.Tensor, graph: dgl.DGLGraph) -> None:
        """Set the provided node features

        Args:
            features (torch.Tensor): Node features
            graph (dgl.DGLGraph): Graph to add the features to
        """
        if not self.add_loc_feats:
            graph.ndata[FEATURES] = features
        elif (
            self.add_loc_feats
            and hasattr(graph, "gdata")
            and "image_size" in graph.gdata.keys()
        ):
            centroids = graph.ndata[CENTROID]
            image_size = graph.gdata["image_size"]
            concat_features = torch.cat(
                (
                    features,
                    (centroids[:, 0].squeeze() / image_size[0]).unsqueeze(1),
                    (centroids[:, 1].squeeze() / image_size[1]).unsqueeze(1),
                ),
                dim=1,
            )
            graph.ndata[FEATURES] = concat_features
        else:
            raise ValueError(
                "Please provide image size to add the normalized centroid to the node features."
            )

    @staticmethod
    @abstractmethod
    def _set_node_centroids(structure: np.ndarray, graph: dgl.DGLGraph) -> None:
        """Set the centroids of the graphs

        Args:
            structure (np.ndarray): Structure of the graph
            graph (dgl.DGLGraph): Graph to add the centroids to
        """

    @abstractmethod
    def _set_node_labels(
        self, structure: np.ndarray, annotation: np.ndarray, graph: dgl.DGLGraph
    ) -> None:
        """Set the node labels of the graphs

        Args:
            structure (np.ndarray): Structure of the graph, eg instance maps, centroids
            annotation (np.ndarray): Annotations, eg node labels
            graph (dgl.DGLGraph): Graph to add the centroids to
        """

    @abstractmethod
    def _build_topology(self, instances: np.ndarray, graph: dgl.DGLGraph) -> None:
        """Generate the graph topology from the provided structure

        Args:
            instances (np.array): Graph structure
            graph (dgl.DGLGraph): Graph to add the edges
        """

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
            self._link_to_path(Path(link_path) / "graphs")


class RAGGraphBuilder(BaseGraphBuilder):
    """
    Super-pixel Graphs class for graph building.
    """

    def __init__(self, kernel_size: int = 5, hops: int = 1, **kwargs) -> None:
        """Create a graph builder that uses a provided kernel size to detect connectivity

        Args:
            kernel_size (int, optional): Size of the kernel to detect connectivity. Defaults to 5.
        """
        logging.debug("*** RAG Graph Builder ***")
        assert hops > 0 and isinstance(
            hops, int
        ), f"Invalid hops {hops} ({type(hops)}). Must be integer >= 0"
        self.kernel_size = kernel_size
        self.hops = hops
        super().__init__(**kwargs)

    @staticmethod
    def _set_node_centroids(instance_map: np.ndarray, graph: dgl.DGLGraph) -> None:
        regions = regionprops(instance_map)
        centroids = np.empty((len(regions), 2))
        for i, region in enumerate(regions):
            center_y, center_x = region.centroid  # row, col
            center_x = int(round(center_x))
            center_y = int(round(center_y))
            centroids[i, 0] = center_x
            centroids[i, 1] = center_y
        graph.ndata[CENTROID] = torch.FloatTensor(centroids)

    def _set_node_labels(
        self, instance_map: np.ndarray, annotation: np.ndarray, graph: dgl.DGLGraph
    ) -> None:
        assert (
            self.nr_classes < 256
        ), "Cannot handle that many classes with 8 byte representation"
        region_labels = pd.unique(np.ravel(instance_map))
        labels = torch.empty(len(region_labels), dtype=torch.uint8)
        for region_label in region_labels:
            histogram = fast_histogram(
                annotation[instance_map == region_label], nr_values=self.nr_classes
            )
            mask = np.ones(len(histogram), np.bool)
            mask[self.background_class] = 0
            if histogram[mask].sum() == 0:
                assignment = self.background_class
            else:
                histogram[self.background_class] = 0
                assignment = np.argmax(histogram)
            labels[region_label - 1] = int(assignment)
        graph.ndata[LABEL] = labels

    def _build_topology(self, instance_map: np.ndarray, graph: dgl.DGLGraph) -> None:
        """Create the graph topology from the connectivty of the provided instance_map

        Args:
            instance_map (np.array): Instance map
            graph (dgl.DGLGraph): Graph to add the edges to
        """
        instance_ids = np.sort(pd.unique(np.ravel(instance_map))).astype(int)
        # background = 0
        if 0 in instance_ids:
            instance_ids = np.delete(instance_ids, np.where(instance_ids == 0))

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        adjacency = np.zeros(shape=(len(instance_ids), len(instance_ids)))
        for instance_id in instance_ids:
            mask = (instance_map == instance_id).astype(np.uint8)
            dilation = cv2.dilate(mask, kernel, iterations=1)
            boundary = dilation - mask
            idx = pd.unique(instance_map[boundary.astype(bool)])
            instance_id -= 1  # because instance_map id starts from 1
            idx -= 1  # because instance_map id starts from 1
            adjacency[instance_id, idx] = 1

        edge_list = np.nonzero(adjacency)
        graph.add_edges(list(edge_list[0]), list(edge_list[1]))

        for _ in range(self.hops - 1):
            graph = two_hop_neighborhood(graph)


class KNNGraphBuilder(BaseGraphBuilder):
    """
    k-Nearest Neighbors Graph class for graph building.
    """

    def __init__(self, k: int = 5, thresh: int = None, **kwargs) -> None:
        """Create a graph builder that uses the (thresholded) kNN algorithm to define the graph topology.

        Args:
            k (int, optional): Number of neighbors. Defaults to 5.
            thresh (int, optional): Maximum allowed distance between 2 nodes. Defaults to None (no thresholding).
        """
        logging.debug("*** kNN Graph Builder ***")
        self.k = k
        self.thresh = thresh
        super().__init__(**kwargs)

    @staticmethod
    def _set_node_centroids(centroids: np.ndarray, graph: dgl.DGLGraph) -> None:
        graph.ndata[CENTROID] = torch.FloatTensor(centroids)

    def _set_node_labels(
        self, instance_map: np.ndarray, annotation: np.ndarray, graph: dgl.DGLGraph
    ) -> None:
        graph.ndata[LABEL] = torch.FloatTensor(annotation.astype(float))

    def _build_topology(self, centroids: np.ndarray, graph: dgl.DGLGraph) -> None:
        """
        Build topology using (thresholded) kNN

        Args:
            centroids (np.array): Centroid locations
            graph (dgl.DGLGraph): Graph to add the edges to
        """

        # build kNN adjacency
        adj = kneighbors_graph(
            centroids, self.k, mode="distance", include_self=False, metric="euclidean"
        ).toarray()

        # filter edges that are too far (ie larger than thresh)
        if self.thresh is not None:
            adj[adj > self.thresh] = 0

        edge_list = np.nonzero(adj)
        graph.add_edges(list(edge_list[0]), list(edge_list[1]))
