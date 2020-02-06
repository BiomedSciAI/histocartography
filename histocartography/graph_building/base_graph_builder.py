import dgl

from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN, CENTROID


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

    def __call__(self, *args):
        """
        Build graph
        Args:
            :param node_features: (FloatTensor) node features
            ;param centroid: (list) centroid of each node normalized by the image size
        """
        if self.config['graph_building_type'] == 'rag_graph_builder':
            graph = self._build_topology(args[2])
            self._set_node_features(args[0], args[1], graph)
        else:
            num_nodes = args[0].shape[0]
            graph = dgl.DGLGraph()
            graph.add_nodes(num_nodes)

            self._set_node_features(args[0], args[1], graph)
            self._build_topology(args[1], graph)
            if self.config['edge_encoding']:
                self._set_edge_embeddings(graph)
        return graph

    def _set_node_features(self, cell_features, centroid, graph):
        """
        Set node embeddings
        """
        graph.ndata[GNN_NODE_FEAT_IN] = cell_features
        graph.ndata[CENTROID] = centroid

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
