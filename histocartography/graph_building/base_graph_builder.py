import dgl

from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN


class BaseGraphBuilder:
    """
    Base interface class for graph building.
    """

    def __init__(self, config, cuda=False, verbose=False):
        """
        Base Graph Builder constructor.

        Args:
            config: (dict) graph building params
            cuda: (bool) if cuda is available
            verbose: (bool) verbosity
        """
        super(BaseGraphBuilder, self).__init__()

        if verbose:
            print('*** Build base graph builder ***')

        self.config = config
        self.cuda = cuda
        self.verbose = verbose

    def __call__(self, node_features, centroid):
        """
        Build graph
        Args:
            :param node_features: (FloatTensor) node features
            ;param centroid: (list) centroid of each node normalized by the image size
        """
        num_nodes = node_features.shape[0]
        graph = dgl.DGLGraph()
        graph.add_nodes(num_nodes)
        self._set_node_features(node_features, graph)
        self._build_topology(centroid, graph)
        if self.config['edge_encoding']:
            self._set_edge_embeddings(graph)
        return graph

    def _set_node_features(self, cell_features, graph):
        """
        Set node embeddings
        """
        graph.ndata[GNN_NODE_FEAT_IN] = cell_features

    def _set_edge_embeddings(self, graph):
        """
        Set edge embedding
        """
        raise NotImplementedError('Edge features are currently not supported.')

    def _build_topology(self, objects):
        """
        Build topology
        :return:
        """
        raise NotImplementedError('Implementation in subclasses.')
