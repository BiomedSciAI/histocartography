import torch
import torch.nn as nn

from histocartography.ml.layers.multi_layer_gnn import MultiLayerGNN
from histocartography.ml.models.base_model import BaseModel
from histocartography.ml.layers.constants import GNN_LL_NODE_FEAT, GNN_NODE_FEAT_IN


class MultiLevelGraphModel(BaseModel):
    """
    Multi-level graph model. The information for grading tumors in WSI lies at different scales. By building 2 graphs,
    one at the cell level and one at the object level (modeled with super pixels), we can extract graph embeddings
    that once combined provide a multi-scale representation of a tumor.

    This implementation is using GIN Layers as a graph neural network and a spatial assignment matrix.

    """

    def __init__(self, config, ll_node_dim, hl_node_dim):
        """
        MultiLevelGraph model constructor
        :param config: (dict) configuration parameters
        :param ll_node_dim: (int) low level node dim, data specific argument
        :param hl_node_dim: (int) high level node dim, data specific argument
        """

        super(MultiLevelGraphModel, self).__init__()

        # 1- set class attributes
        self.config = config
        self.ll_node_dim = ll_node_dim
        self.hl_node_dim = hl_node_dim

        self.num_classes = config['num_classes']
        self.dropout = config['dropout']
        self.use_bn = config['use_bn']
        self.concat = config['cat']

        # 2- build cell graph params
        self._build_cell_graph_params()

        # 3- build super pixel graph params
        self._build_superpx_graph_params()

        # 4- build classification params
        self._build_classification_params()

    def _build_cell_graph_params(self):
        """
        Build cell graph multi layer GNN
        """
        self._update_config(self.config['gnn_params'][0], self.ll_node_dim)
        self.cell_graph_gnn = MultiLayerGNN(config=self.config['gnn_params'][0])

    def _build_superpx_graph_params(self):
        """
        Build super pixel multi layer GNN
        """
        self._update_config(self.config['gnn_params'][1], self.hl_node_dim + self.config['gnn_params'][0]['output_dim'])
        self.superpx_gnn = MultiLayerGNN(config=self.config['gnn_params'][1])

    def _build_classification_params(self):
        """
        Build classification parameters
        """
        if self.concat:
            hidden_dim = self.config['gnn_params'][0]['input_dim'] + \
                self.config['gnn_params'][0]['hidden_dim'] * (self.config['gnn_params'][0]['n_layers'] - 1) + \
                self.config['gnn_params'][0]['output_dim'] + \
                self.config['gnn_params'][1]['input_dim'] + \
                self.config['gnn_params'][1]['hidden_dim'] * (self.config['gnn_params'][1]['n_layers'] - 1) + \
                self.config['gnn_params'][1]['output_dim']
        else:
            hidden_dim = self.config['gnn_params'][-1]['output_dim']
        self.pred_layer = nn.Linear(hidden_dim, self.num_classes)

    def _update_config(self, config, input_dim=None):
        """
        Update config params with data-dependent parameters
        """
        if input_dim is not None:
            config['input_dim'] = input_dim

        config['use_bn'] = self.use_bn

    def _compute_assigned_feats(self, graph, feats, assignment):
        """
        Use the assignment matrix to agg the feats
        :param graph: (DGLBatch)
        :param feats: (FloatTensor)
        :param assignment: (list of LongTensor)
        """
        num_nodes_per_graph = graph.batch_num_nodes
        num_nodes_per_graph.insert(0, 0)
        intervals = [sum(num_nodes_per_graph[:i+1]) for i in range(len(num_nodes_per_graph))]

        ll_h_concat = []
        for i in range(1, len(intervals)):
            sum_ = torch.matmul(assignment[i-1], feats[intervals[i-1]:intervals[i], :])
            ll_h_concat.append(sum_)

        return torch.cat(ll_h_concat, dim=0)

    def forward(self, cell_graph, superpx_graph, assignment_matrix):
        """
        Foward pass.
        :param cell_graph: (DGLGraph) low level graph
        :param superpx_graph: (DGLGraph) high level graph
        :param assignment_matrix: (list of LongTensor) define how to pool
                                  the low level graph to build high level
                                  features.
        """
        # 1. GNN layers over the low level graph
        ll_feats = cell_graph.ndata[GNN_NODE_FEAT_IN]
        ll_h = self.cell_graph_gnn(cell_graph, ll_feats, self.concat)
        cell_graph.ndata[GNN_LL_NODE_FEAT] = ll_h

        # 2. Sum the low level features according to assignment matrix
        ll_h_concat = self._compute_assigned_feats(cell_graph, ll_h, assignment_matrix)

        superpx_graph.ndata[GNN_NODE_FEAT_IN] = torch.cat((ll_h_concat, superpx_graph.ndata[GNN_NODE_FEAT_IN]), dim=1)

        # 3. GNN layers over the high level graph
        hl_feats = superpx_graph.ndata[GNN_NODE_FEAT_IN]
        hl_h = self.superpx_gnn(superpx_graph, hl_feats, self.concat)

        # 4. Classification layers
        if self.concat:
            graph_embeddings = torch.cat((ll_h, hl_h), dim=0)
        else:
            graph_embeddings = hl_h

        logits = self.pred_layer(graph_embeddings)
        return logits
