import torch

from histocartography.ml.models.base_model import BaseModel
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN, READOUT_TYPES, GNN_NODE_FEAT_OUT
from histocartography.ml.layers.mlp import MLP


class MultiLevelGraphModel(BaseModel):
    """
    Multi-level graph model. The information for grading tumors in WSI lies at different scales. By building 2 graphs,
    one at the cell level and one at the object level (modeled with super pixels), we can extract graph embeddings
    that once combined provide a multi-scale representation of a tumor.

    This implementation is using GIN Layers as a graph neural network and a spatial assignment matrix.

    """

    def __init__(self, config, node_dims):
        """
        MultiLevelGraph model constructor
        :param config: (dict) configuration parameters
        :param ll_node_dim: (int) low level node dim, data specific argument
        :param hl_node_dim: (int) high level node dim, data specific argument
        """

        super(MultiLevelGraphModel, self).__init__(config)

        # 1- set class attributes
        self.config = config
        self.ll_node_dim = node_dims[0]
        self.hl_node_dim = node_dims[1]
        self.cell_gnn_params = config['gnn_params']['cell_gnn']
        self.superpx_gnn_params = config['gnn_params']['superpx_gnn']
        self.readout_params = self.config['readout']

        # 2- build cell graph params
        self._build_cell_graph_params(self.cell_gnn_params)

        # 3- build super pixel graph params
        if self.concat:
            superpx_input_dim = self.hl_node_dim +\
                                self.cell_gnn_params['output_dim'] +\
                                self.cell_gnn_params['hidden_dim'] * (self.cell_gnn_params['n_layers'] - 1) +\
                                self.cell_gnn_params['input_dim']
        else:
            superpx_input_dim = self.hl_node_dim + self.cell_gnn_params['output_dim']
        self._build_superpx_graph_params(
            self.superpx_gnn_params,
            input_dim=superpx_input_dim
        )

        # 4- build classification params
        self._build_classification_params()

    def _build_classification_params(self):
        """
        Build classification parameters
        """
        if self.concat:
            emd_dim = self.superpx_gnn_params['input_dim'] + \
                self.superpx_gnn_params['hidden_dim'] * (self.superpx_gnn_params['n_layers'] - 1) + \
                self.superpx_gnn_params['output_dim']
        else:
            emd_dim = self.superpx_gnn_params['output_dim']

        self.pred_layer = MLP(in_dim=emd_dim,
                              h_dim=self.readout_params['hidden_dim'],
                              out_dim=self.num_classes,
                              num_layers=self.readout_params['num_layers']
                              )

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
        intervals = [sum(num_nodes_per_graph[:i + 1])
                     for i in range(len(num_nodes_per_graph))]

        ll_h_concat = []

        for i in range(1, len(intervals)):
            sum_ = torch.matmul(
                assignment[i - 1], feats[intervals[i - 1]:intervals[i], :]
            )
            ll_h_concat.append(sum_)

        return torch.cat(ll_h_concat, dim=0)

    def forward(self, data):
        """
        Foward pass.
        :param data: tuple of (DGLGraph) low level graph,
                                (DGLGraph) high level graph,
                                (list of LongTensor) define how to pool
                                the low level graph to build high level
                                features.
        """

        cell_graph, superpx_graph, assignment_matrix = data[0], data[1], data[2]

        # 1. GNN layers over the low level graph
        ll_feats = cell_graph.ndata[GNN_NODE_FEAT_IN]
        ll_h = self.cell_graph_gnn(cell_graph, ll_feats, self.concat, with_readout=False)

        # 2. Sum the low level features according to assignment matrix
        ll_h_concat = self._compute_assigned_feats(
            cell_graph, ll_h, assignment_matrix)

        superpx_graph.ndata[GNN_NODE_FEAT_IN] = torch.cat(
            (ll_h_concat, superpx_graph.ndata[GNN_NODE_FEAT_IN]), dim=1)

        # 3. GNN layers over the high level graph
        hl_feats = superpx_graph.ndata[GNN_NODE_FEAT_IN]
        graph_embeddings = self.superpx_gnn(superpx_graph, hl_feats, self.concat)

        # 4. Classification layers
        logits = self.pred_layer(graph_embeddings)
        return logits
