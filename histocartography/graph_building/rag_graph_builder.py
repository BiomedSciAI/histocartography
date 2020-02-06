from histocartography.graph_building.base_graph_builder import BaseGraphBuilder
from dgl.data.utils import load_graphs


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

    def _build_topology(self, dgl_graph_file):
        """
        Loads the dgl graph from file
        """

        g, label_dict = load_graphs(dgl_graph_file)

        return g[0]

