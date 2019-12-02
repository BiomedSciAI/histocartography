import itertools
import torch
import dgl

from histocartography.graph_building.base_graph_builder import BaseGraphBuilder
from histocartography.graph_building.constants import LABEL, VISUAL, CENTROID
from histocartography.utils.vector import compute_l2_distance, compute_edge_weight


class WaxmanGraphBuilder(BaseGraphBuilder):
    """
    Waxman class for graph building.

    Current implementation is deterministic, ie the edge weights is determistically assigned
    to the graph.

    """

    def __init__(self, config, cuda=False, verbose=False):
        """
        Waxman Graph Builder constructor.

        Args:
            config: list of required params to build a graph
            cuda: (bool) if cuda is available
            verbose: (bool) verbosity level
        """
        super(WaxmanGraphBuilder, self).__init__(config, cuda, verbose)

        if verbose:
            print('*** Build Waxman graph ***')

        self.config = config
        self.cuda = cuda

    def __call__(self, objects, image_size):
        """
        Build graph
        Args:
            objects: (list) each element in the list is a dict with:
                - centroid
                - label
                - visual descriptor
            image_size: (list) weight and height of the image
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
        graph.ndata[CENTROID] = torch.LongTensor([obj[CENTROID] for obj in objects])
        graph.ndata[LABEL] = torch.LongTensor([obj[LABEL] for obj in objects])
        # graph.ndata[VISUAL] = torch.LongTensor([obj[VISUAL] for obj in objects])

    def _set_edge_embeddings(self, objects, graph):
        """
        Build edge embedding
        """

    def _build_topology(self, objects, graph):
        """
        Build topology.
        """
        num_objects = len(objects)
        centroid = [obj[CENTROID] for obj in objects]
        src = []
        dst = []
        for pair in itertools.combinations(range(num_objects), 2):
            dist = compute_l2_distance(centroid[pair[0]], centroid[pair[1]])
            edge_weight = compute_edge_weight(dist)
            if edge_weight > self.config['edge_threshold']:

                # src -> dst
                src.append(pair[0])
                dst.append(pair[1])

                # dst -> src
                src.append(pair[1])
                dst.append(pair[0])

        graph.add_edges(src, dst)

