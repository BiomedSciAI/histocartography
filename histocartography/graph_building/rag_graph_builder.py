from histocartography.graph_building.base_graph_builder import BaseGraphBuilder
from skimage.future import graph
import numpy as np
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN, CENTROID


class RAGGraphBuilder(BaseGraphBuilder):
    """
    RAG(Region adjacency Graph) class for graph building.
    """

    def __init__(self, config, cuda=False, verbose=False):
        """
        RA Graph Builder constructor.

        Args:
            config: list of required params to build a graph
            cuda: (bool) if cuda is available
            verbose: (bool) verbosity level
        """
        super(RAGGraphBuilder, self).__init__(config, cuda, verbose)

        if verbose:
            print('*** Build region adjacency graph ***')

        self.config = config
        self.cuda = cuda

    def _build_topology(self, sp_map):

        """
        Returns the RAG in form of networkx graph
        """
        g = graph.RAG(sp_map, connectivity=2)

        return g

