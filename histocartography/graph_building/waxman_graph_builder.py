import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from histocartography.graph_building.base_graph_builder import BaseGraphBuilder


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

    def _build_topology(self, centroid, graph):
        """
        Build topology using the distance between the centroids of each node.
            If the distance is smaller than a threshold, then we build an edge.
        """
        out = euclidean_distances(centroid.numpy())
        src, dst = np.nonzero(out < self.config['edge_threshold'])
        graph.add_edges(src, dst)
