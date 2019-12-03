import torch
import dgl
import numpy as np
from sklearn.neighbors import kneighbors_graph

from histocartography.graph_building.base_graph_builder import BaseGraphBuilder
from histocartography.graph_building.constants import LABEL, CENTROID


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

    def _build_topology(self, objects, graph):
        """
        Build topology
        """
        centroid = [obj[CENTROID] for obj in objects]

        # build adjacency matrix
        adj = kneighbors_graph(
            centroid,
            self.config['n_neighbors'],
            mode='distance',
            include_self=False,
            metric='euclidean'
        )
        adj = np.clip(adj.toarray(), self.config['edge_threshold'], 1)

        # append edges
        edge_list = np.nonzero(adj)
        graph.add_edges(list(edge_list[0]), list(edge_list[1]))

