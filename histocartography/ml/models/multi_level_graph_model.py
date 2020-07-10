import torch
import torch.nn as nn
import torch.nn.functional as F

from histocartography.ml.models.base_model import BaseModel
from histocartography.ml.layers.constants import (
    GNN_NODE_FEAT_IN, READOUT_TYPES,
    GNN_NODE_FEAT_OUT, AGGREGATORS,
    SCALERS
    )   
from histocartography.ml.layers.mlp import MLP

# debug purposes 
import time 


class MultiLevelGraphModel(BaseModel):
    """
    Multi-level graph model. The information for grading tumors in WSI lies at different scales. By building 2 graphs,
    one at the cell level and one at the object level (modeled with super pixels), we can extract graph embeddings
    that once combined provide a multi-scale representation of a tumor.

    This implementation is using GIN Layers as a graph neural network and a spatial assignment matrix.

    """

    def __init__(self, config, input_feature_dims):
        """
        MultiLevelGraph model constructor
        :param config: (dict) configuration parameters
        :param ll_node_dim: (int) low level node dim, data specific argument
        :param hl_node_dim: (int) high level node dim, data specific argument
        """

        super(MultiLevelGraphModel, self).__init__(config)

        # 1- set class attributes
        self.config = config
        self.ll_node_dim, self.hl_node_dim, self.edge_dim, _ = input_feature_dims
        self.cell_gnn_params = config['gnn_params']['cell_gnn']
        self.superpx_gnn_params = config['gnn_params']['superpx_gnn']
        self.readout_params = self.config['readout']
        self.readout_agg_op = config['gnn_params']['cell_gnn']['agg_operator']
        self.pna_assignment = config['gnn_params']['cell_gnn']['layer_type'] == 'pna_layer'
        self.pna_assignment = False

        # 2- build cell graph params
        self._build_cell_graph_params(self.cell_gnn_params)

        # 3- build super pixel graph params
        if self.readout_agg_op == "concat" and not self.pna_assignment:
            superpx_input_dim = self.hl_node_dim +\
                                self.cell_gnn_params['output_dim'] +\
                                self.cell_gnn_params['hidden_dim'] * (self.cell_gnn_params['n_layers'] - 1) #  +\
                                # self.cell_gnn_params['input_dim']
        else:
            superpx_input_dim = self.hl_node_dim + self.cell_gnn_params['output_dim']
        self._build_superpx_graph_params(
            self.superpx_gnn_params,
            input_dim=superpx_input_dim
        )

        # 4- build assignement operator(s)
        if self.pna_assignment:
            self.aggregators = [AGGREGATORS[aggr] for aggr in config['gnn_params']['cell_gnn']['aggregators'].split()]
            self.scalers = [SCALERS[scale] for scale in config['gnn_params']['cell_gnn']['scalers'].split()]
            in_dim = (len(self.aggregators) * len(self.scalers)) * self.cell_gnn_params['output_dim']
            if self.readout_agg_op == "concat":
                in_dim *= self.cell_gnn_params['n_layers']
            self.assignment_mapper = nn.Sequential(
                MLP(
                    in_dim=in_dim,
                    h_dim=self.cell_gnn_params['output_dim'],
                    out_dim=self.cell_gnn_params['output_dim'],
                    num_layers=1,
                    act='relu'
                ),
                nn.ReLU()
            )
            self.avg_d = {"log": 3.3}

        # 4- build classification params
        self._build_classification_params()

    def _build_classification_params(self):
        """
        Build classification parameters
        """
        if self.readout_agg_op == "concat":
            emd_dim = self.superpx_gnn_params['hidden_dim'] * (self.superpx_gnn_params['n_layers'] - 1) + \
                self.superpx_gnn_params['output_dim']
        else:
            emd_dim = self.superpx_gnn_params['output_dim']

        self.pred_layer = MLP(in_dim=emd_dim,
                              h_dim=self.readout_params['hidden_dim'],
                              out_dim=self.num_classes,
                              num_layers=self.readout_params['num_layers']
                              )

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
            if self.pna_assignment:
                h_agg = []
                for row_idx in range(assignment[i - 1].shape[0]):
                    subidx = (assignment[i - 1][row_idx, :] != 0).nonzero().squeeze(dim=1)
                    subfeats = feats[intervals[i - 1]:intervals[i], :][subidx, :].unsqueeze(dim=0)
                    degree = subfeats.shape[1]
                    if degree > 0:
                        h_agg_row = torch.cat([aggregate(subfeats) for aggregate in self.aggregators], dim=1)
                        h_agg_row = torch.cat([scale(h_agg_row, D=degree, avg_d=self.avg_d) for scale in self.scalers], dim=1).squeeze()
                    else:
                        h_agg_row = torch.zeros(len(self.aggregators) * len(self.scalers) * subfeats.shape[-1]).to(feats.get_device())
                    h_agg.append(h_agg_row.unsqueeze(dim=0))
                h_agg = torch.cat(h_agg, dim=0)
            else:
                h_agg = torch.matmul(
                    assignment[i - 1], feats[intervals[i - 1]:intervals[i], :]
                )

            if self.pna_assignment:
                h_agg = self.assignment_mapper(h_agg)

            ll_h_concat.append(h_agg)

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
        ll_h = self.cell_graph_gnn(cell_graph, ll_feats, with_readout=False)

        # 2. Sum the low level features according to assignment matrix
        ll_h_concat = self._compute_assigned_feats(
            cell_graph, ll_h, assignment_matrix)

        superpx_graph.ndata[GNN_NODE_FEAT_IN] = torch.cat(
            (ll_h_concat, superpx_graph.ndata[GNN_NODE_FEAT_IN]), dim=1)

        # 3. GNN layers over the high level graph
        hl_feats = superpx_graph.ndata[GNN_NODE_FEAT_IN]
        graph_embeddings = self.superpx_gnn(superpx_graph, hl_feats)

        # 4. Classification layers
        logits = self.pred_layer(graph_embeddings)
        return logits
