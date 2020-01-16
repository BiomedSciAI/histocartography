import numpy as np
from sklearn.neighbors import kneighbors_graph

from histocartography.graph_building.base_graph_builder import BaseGraphBuilder
from histocartography.utils.vector import compute_l2_distance


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
        # filter edges that are too far apart
        edge_list = np.nonzero(adj)
        edge_list = [[s, d] for (s, d) in zip(edge_list[0], edge_list[1])]
        edge_list = list(
            filter(
                lambda x: compute_l2_distance(centroid[x[0]], centroid[x[1]]) < self.config['max_distance'], edge_list
            )
        )
        # append edges
        edge_list = ([s for (s, _) in edge_list], [d for (_, d) in edge_list])
        graph.add_edges(list(edge_list[0]), list(edge_list[1]))
