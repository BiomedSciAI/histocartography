import dgl
import torch

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
            verbose: (bool) verbosity level
        """
        super(BaseGraphBuilder, self).__init__()

        if verbose:
            print('*** Build base graph builder ***')

        self.config = config
        self.cuda = cuda
        self.verbose = verbose

    def __call__(self, cell_features, centroid):
        """
        Build graph
        Args:
            objects: (list) each element in the list is a dict with:
                - centroid
                - label
                - visual descriptor
            image_size: (list) weight and height of the image
        """
        num_nodes = cell_features.shape[0]
        graph = dgl.DGLGraph()
        graph.add_nodes(num_nodes)
        self._set_node_features(cell_features, graph)
        self._build_topology(centroid, graph)
        if self.config['edge_encoding']:
            self._set_edge_embeddings(graph)
        return graph

    def _set_node_features(self, cell_features, graph):
        """
        Build node embeddings
        """
        graph.ndata[GNN_NODE_FEAT_IN] = cell_features

    def _set_edge_embeddings(self, graph):
        """
        Build edge embedding
        """

    def _build_topology(self, objects):
        """
        Build topology
        :return:
        """