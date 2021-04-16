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

from ..pipeline import PipelineStep
from .utils import fast_histogram


LABEL = "label"
CENTROID = "centroid"
FEATURES = "feat"


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
            self,
            nr_annotation_classes: int = 5,
            annotation_background_class: Optional[int] = None,
            add_loc_feats: bool = False,
            **kwargs: Any
    ) -> None:
        """
        Base Graph Builder constructor.
        Args:
            nr_annotation_classes (int): Number of classes in annotation. Used only if setting node labels.
            annotation_background_class (int): Background class label in annotation. Used only if setting node labels.
            add_loc_feats (bool): Flag to include location-based features (ie normalized centroids)
                                  in node feature representation.
                                  Defaults to False.
        """
        self.nr_annotation_classes = nr_annotation_classes
        self.annotation_background_class = annotation_background_class
        self.add_loc_feats = add_loc_feats
        super().__init__(**kwargs)

    def _process(  # type: ignore[override]
        self,
        instance_map: np.ndarray,
        features: torch.Tensor,
        annotation: Optional[np.ndarray] = None,
    ) -> dgl.DGLGraph:
        """Generates a graph from a given instance_map and features
        Args:
            instance_map (np.array): Instance map depicting tissue components
            features (torch.Tensor): Features of each node. Shape (nr_nodes, nr_features)
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
        image_size = (instance_map.shape[1], instance_map.shape[0])  # (x, y)

        # get instance centroids
        centroids = self._get_node_centroids(instance_map)

        # add node content
        self._set_node_centroids(centroids, graph)
        self._set_node_features(features, image_size, graph)
        if annotation is not None:
            self._set_node_labels(instance_map, annotation, graph)

        # build edges
        self._build_topology(instance_map, centroids, graph)
        return graph

    def _process_and_save(  # type: ignore[override]
        self,
        instance_map: np.ndarray,
        features: torch.Tensor,
        annotation: Optional[np.ndarray] = None,
        output_name: str = None,
    ) -> dgl.DGLGraph:
        """Process and save in provided directory
        Args:
            output_name (str): Name of output file
            instance_map (np.ndarray): Instance map depicting tissue components
                                       (eg nuclei, tissue superpixels)
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
                instance_map=instance_map,
                features=features,
                annotation=annotation)
            save_graphs(str(output_path), [graph])
        return graph

    def _get_node_centroids(
            self, instance_map: np.ndarray
    ) -> np.ndarray:
        """Get the centroids of the graphs
        Args:
            instance_map (np.ndarray): Instance map depicting tissue components
        Returns:
            centroids (np.ndarray): Node centroids
        """
        regions = regionprops(instance_map)
        centroids = np.empty((len(regions), 2))
        for i, region in enumerate(regions):
            center_y, center_x = region.centroid  # (y, x)
            center_x = int(round(center_x))
            center_y = int(round(center_y))
            centroids[i, 0] = center_x
            centroids[i, 1] = center_y
        return centroids

    def _set_node_centroids(
            self,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Set the centroids of the graphs
        Args:
            centroids (np.ndarray): Node centroids
            graph (dgl.DGLGraph): Graph to add the centroids to
        """
        graph.ndata[CENTROID] = torch.FloatTensor(centroids)

    def _set_node_features(
            self,
            features: torch.Tensor,
            image_size: Tuple[int, int],
            graph: dgl.DGLGraph
    ) -> None:
        """Set the provided node features

        Args:
            features (torch.Tensor): Node features
            image_size (Tuple[int,int]): Image dimension (x, y)
            graph (dgl.DGLGraph): Graph to add the features to
        """
        if not torch.is_tensor(features):
            features = torch.FloatTensor(features)
        if not self.add_loc_feats:
            graph.ndata[FEATURES] = features
        elif (
                self.add_loc_feats
                and image_size is not None
        ):
            # compute normalized centroid features
            centroids = graph.ndata[CENTROID]

            normalized_centroids = torch.empty_like(centroids)  # (x, y)
            normalized_centroids[:, 0] = centroids[:, 0] / image_size[0]
            normalized_centroids[:, 1] = centroids[:, 1] / image_size[1]

            if features.ndim == 3:
                normalized_centroids = normalized_centroids \
                    .unsqueeze(dim=1) \
                    .repeat(1, features.shape[1], 1)
                concat_dim = 2
            elif features.ndim == 2:
                concat_dim = 1

            concat_features = torch.cat(
                (
                    features,
                    normalized_centroids
                ),
                dim=concat_dim,
            )
            graph.ndata[FEATURES] = concat_features
        else:
            raise ValueError(
                "Please provide image size to add the normalized centroid to the node features."
            )

    @abstractmethod
    def _set_node_labels(
            self,
            instance_map: np.ndarray,
            annotation: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Set the node labels of the graphs
        Args:
            instance_map (np.ndarray): Instance map depicting tissue components
            annotation (np.ndarray): Annotations, eg node labels
            graph (dgl.DGLGraph): Graph to add the centroids to
        """

    @abstractmethod
    def _build_topology(
            self,
            instance_map: np.ndarray,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Generate the graph topology from the provided instance_map
        Args:
            instance_map (np.array): Instance map depicting tissue components
            centroids (np.array): Node centroids
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

    def __init__(self, kernel_size: int = 3, hops: int = 1, **kwargs) -> None:
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

    def _set_node_labels(
            self,
            instance_map: np.ndarray,
            annotation: np.ndarray,
            graph: dgl.DGLGraph) -> None:
        """Set the node labels of the graphs using annotation map"""
        assert (
            self.nr_annotation_classes < 256
        ), "Cannot handle that many classes with 8-bits"
        regions = regionprops(instance_map)
        labels = torch.empty(len(regions), dtype=torch.uint8)

        for region_label in np.arange(1, len(regions) + 1):
            histogram = fast_histogram(
                annotation[instance_map == region_label],
                nr_values=self.nr_annotation_classes
            )
            mask = np.ones(len(histogram), np.bool)
            mask[self.annotation_background_class] = 0
            if histogram[mask].sum() == 0:
                assignment = self.annotation_background_class
            else:
                histogram[self.annotation_background_class] = 0
                assignment = np.argmax(histogram)
            labels[region_label - 1] = int(assignment)
        graph.ndata[LABEL] = labels

    def _build_topology(
            self,
            instance_map: np.ndarray,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Create the graph topology from the instance connectivty in the instance_map"""
        regions = regionprops(instance_map)
        instance_ids = torch.empty(len(regions), dtype=torch.uint8)

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        adjacency = np.zeros(shape=(len(instance_ids), len(instance_ids)))

        for instance_id in np.arange(1, len(instance_ids) + 1):
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

    def _set_node_labels(
            self,
            instance_map: np.ndarray,
            annotation: np.ndarray,
            graph: dgl.DGLGraph) -> None:
        """Set the node labels of the graphs using annotation"""
        regions = regionprops(instance_map)
        assert annotation.shape[0] == len(regions), \
            "Number of annotations do not match number of nodes"
        graph.ndata[LABEL] = torch.FloatTensor(annotation.astype(float))

    def _build_topology(
            self,
            instance_map: np.ndarray,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Build topology using (thresholded) kNN"""

        # build kNN adjacency
        adj = kneighbors_graph(
            centroids,
            self.k,
            mode="distance",
            include_self=False,
            metric="euclidean").toarray()

        # filter edges that are too far (ie larger than thresh)
        if self.thresh is not None:
            adj[adj > self.thresh] = 0

        edge_list = np.nonzero(adj)
        graph.add_edges(list(edge_list[0]), list(edge_list[1]))
