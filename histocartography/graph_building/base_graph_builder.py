import dgl
import torch 
import time 
import numpy as np
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN, CENTROID, GNN_EDGE_FEAT


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
            # self._set_node_features(args[0], args[1], graph)
        else:
            num_nodes = args[0].shape[0]
            graph = dgl.DGLGraph()
            graph.add_nodes(num_nodes)
            self._set_node_features(args[0], args[1], graph)
            self._build_topology(args[1], graph)

        # add optional edge features 
        if self.config['edge_encoding']:
            self._set_edge_embeddings(args[1], args[2], graph)
        return graph

    def _set_node_features(self, cell_features, centroid, graph):
        """
        Set node embeddings
        """
        graph.ndata[GNN_NODE_FEAT_IN] = cell_features
        graph.ndata[CENTROID] = centroid

    def _set_edge_embeddings(self, centroids, bounding_boxes, graph):
        """
        Set edge embedding
        """
        # t0 = time.time()
        # efeat = torch.FloatTensor([[
        #     (centroids[from_][0] - centroids[to_][0]) / (bounding_boxes[to_][2] - bounding_boxes[to_][0]),
        #     (centroids[from_][1] - centroids[to_][1]) / (bounding_boxes[to_][3] - bounding_boxes[to_][1]),
        #     torch.log((bounding_boxes[from_][2] - bounding_boxes[from_][0]) / (bounding_boxes[to_][2] - bounding_boxes[to_][0])),
        #     torch.log((bounding_boxes[from_][3] - bounding_boxes[from_][1]) / (bounding_boxes[to_][3] - bounding_boxes[to_][1]))
        #     ] for from_, to_ in zip(graph.edges()[0], graph.edges()[1])])
        # print('For loop', time.time() - t0)

        # test with numpy 
        edges = torch.cat([graph.edges()[0].unsqueeze(dim=0), graph.edges()[1].unsqueeze(dim=0)]).detach().numpy()
        def encode_edge(row):
            from_, to_ = row
            return np.array([
                (centroids[from_][0] - centroids[to_][0]) / (bounding_boxes[to_][2] - bounding_boxes[to_][0]),
                (centroids[from_][1] - centroids[to_][1]) / (bounding_boxes[to_][3] - bounding_boxes[to_][1]),
                torch.log((bounding_boxes[from_][2] - bounding_boxes[from_][0]) / (bounding_boxes[to_][2] - bounding_boxes[to_][0])),
                torch.log((bounding_boxes[from_][3] - bounding_boxes[from_][1]) / (bounding_boxes[to_][3] - bounding_boxes[to_][1]))])
        efeats = torch.FloatTensor(np.apply_along_axis(encode_edge, 0, edges)).t()
        graph.edata[GNN_EDGE_FEAT] = efeats

    def _build_topology(self, objects):
        """
        Build topology
        :return:
        """
        raise NotImplementedError('Implementation in subclasses.')
