import itertools

from histocartography.graph_building.base_graph_builder import BaseGraphBuilder
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

    def _build_topology(self, centroid, graph):
        """
        Build topology.
        """
        num_objects = len(centroid)
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
