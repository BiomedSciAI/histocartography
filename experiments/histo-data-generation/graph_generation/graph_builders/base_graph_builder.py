import dgl
import torch
import numpy as np
from skimage.measure import regionprops

from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN, GNN_EDGE_FEAT, CENTROID


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

        # add nodes 
        num_nodes = args[0].shape[0]
        graph = dgl.DGLGraph()
        graph.add_nodes(num_nodes)

        # add node features 
        self._set_node_features(args[0], args[1], graph)

        # build edges 
        self._build_topology(args[2 if self.config['graph_building_type'] == 'rag_graph_builder' else 1], graph)

        # add edge features 
        if self.config['edge_encoding']:
            self._set_edge_embeddings(graph, args[2])
        return graph

    def _set_node_features(self, features, centroid, graph):
        """
        Set node embeddings
        """
        graph.ndata[GNN_NODE_FEAT_IN] = features
        graph.ndata[CENTROID] = centroid

    def _set_edge_embeddings(self, graph, instance_map):
        """
        Set edge embedding
        """
        centroids = graph.ndata[CENTROID]
        bounding_boxes = self._extract_instance_bounding_boxes(instance_map.cpu().to(torch.int64).detach().numpy())

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

    def _extract_instance_bounding_boxes(self, instance_map):
        regions = regionprops(instance_map)
        bounding_boxes = []
        for region in regions:
            y1, x1, y2, x2 = region['bbox']
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= instance_map.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= instance_map.shape[0] - 1 else y2
            bounding_box = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            bounding_boxes.append(bounding_box)
        return torch.FloatTensor(bounding_boxes)

    def _build_topology(self, objects):
        """
        Build topology
        :return:
        """
        raise NotImplementedError('Implementation in subclasses.')

