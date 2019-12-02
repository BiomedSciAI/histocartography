import dgl
import torch

from histocartography.graph_building.constants import LABEL, CENTROID


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

    def __call__(self, objects, image_size):
        """
        Build graph
        Args:
            objects: (list) each element in the list is a dict with:
                - centroid
                - label
                - visual descriptor
            image_size: (list) weight and height of the image
        """
        num_objects = len(objects)
        graph = dgl.DGLGraph()
        graph.add_nodes(num_objects)
        self._set_node_features(objects, graph)
        self._build_topology(objects, graph)
        if self.config['edge_encoding']:
            self._set_edge_embeddings(objects, graph)
        return graph

    def _set_node_features(self, objects, graph):
        """
        Build node embeddings
        """
        graph.ndata[CENTROID] = torch.LongTensor([obj[CENTROID] for obj in objects])
        graph.ndata[LABEL] = torch.LongTensor([obj[LABEL] for obj in objects])
        # graph.ndata[VISUAL] = torch.LongTensor([obj[VISUAL] for obj in objects])

    def _set_edge_embeddings(self, objects, graph):
        """
        Build edge embedding
        """

    def _build_topology(self, objects):
        """
        Build topology
        :return:
        """