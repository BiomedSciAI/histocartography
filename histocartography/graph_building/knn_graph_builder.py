import numpy as np
from sklearn.neighbors import kneighbors_graph

from histocartography.graph_building.base_graph_builder import BaseGraphBuilder


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

    def _build_topology(self, centroid, graph):
        """
        Build topology using a kNN algorithm based on the euclidean
            distance between the centroid of the nodes.
        """

        # build adjacency matrix
        adj = kneighbors_graph(
            centroid,
            self.config['n_neighbors'],
            mode='distance',
            include_self=False,
            metric='euclidean'
        )
        # append edges
        edge_list = np.nonzero(adj)
        graph.add_edges(list(edge_list[0]), list(edge_list[1]))
