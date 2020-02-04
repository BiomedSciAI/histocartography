import itertools
from scipy.spatial import Delaunay
from histocartography.graph_building.base_graph_builder import BaseGraphBuilder


class DelaunayGraphBuilder(BaseGraphBuilder):
    """
    Delaunay class for graph building.
    """

    def __init__(self, config, cuda=False, verbose=False):
        """
        Delaunay Graph Builder constructor.

        Args:
            config: list of required params to build a graph
            cuda: (bool) if cuda is available
            verbose: (bool) verbosity level
        """
        super(DelaunayGraphBuilder, self).__init__(config, cuda, verbose)

        if verbose:
            print('*** Build delaunay graph ***')

        self.config = config
        self.cuda = cuda

    def _build_topology(self, centroid, graph):

        tri = Delaunay(centroid)
        edges = []

        # create edges
        for v in tri.simplices:
            for indices in itertools.combinations(v, 2):
                edges.append(indices)
        edge_list = ([s for (s, _) in edges], [d for (_, d) in edges])
        graph.add_edges(list(edge_list[0]), list(edge_list[1]))