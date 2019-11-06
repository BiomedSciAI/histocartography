import itertools
import torch
import dgl

from histocartography.graph_building.base_graph_builder import BaseGraphBuilder
from histocartography.graph_building.constants import LABEL, VISUAL, BOX
from histocartography.utils.vector import compute_box_centroid, compute_l2_distance, compute_edge_weight


class KNNGraphBuilder(BaseGraphBuilder):
    """
    KNN (K-Nearest Neighbors) class for graph building.
    """

    def __init__(self, config, cuda=False, verbose=False):
        """
        k-NN Graph Builder constructor.

        Args:
            config: list of required params to build a graph
            cuda: (bool) if cuda is available
            verbose: (bool) verbosity level
        """
        super(KNNGraphBuilder, self).__init__(config, cuda, verbose)

        if verbose:
            print('*** Build k-NN graph ***')

        self.config = config
        self.cuda = cuda

    def __call__(self, objects):
        """
        Build graph
        Args:
            objects: (list) each element in the list is a dict with:
                - bbox
                - label
                - visual descriptor
        """
        num_objects = len(objects)
        graph = dgl.DGLGraph()
        graph.add_nodes(num_objects)
        self._set_node_features(objects, graph)
        self._build_topology(objects, graph)
        if self.config['edge_encoding']:
            self._set_edge_embeddings(objects, graph)
        return graph

    def _set_node_features(self, objects, graph):
        """
        Build node embeddings
        """
        graph.ndata[BOX] = torch.LongTensor([obj[BOX] for obj in objects])  # @TODO do box normalization ?
        graph.ndata[LABEL] = torch.LongTensor([obj[LABEL] for obj in objects])
        graph.ndata[VISUAL] = torch.LongTensor([obj[VISUAL] for obj in objects])

    def _set_edge_embeddings(self, objects, graph):
        """
        Build edge embedding
        """

    def _build_topology(self, objects, graph):
        """
        Build topology
        """
        num_objects = len(objects)
        boxes = [obj[BOX] for obj in objects]
        centroid = [compute_box_centroid(box) for box in boxes]
        edge_list = []
        for pair in itertools.combination(range(num_objects), 2):
            dist = compute_l2_distance(centroid[pair[0], pair[1]])
            edge_weight = compute_edge_weight(dist)
            if edge_weight > self.config['edge_threshold']:
                edge_list.append(pair)
                edge_list.append([pair[1], pair[0]])
        graph.add_edges(edge_list)

