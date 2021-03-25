from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl 

from .base_model import BaseModel
from ..layers.constants import (
    GNN_NODE_FEAT_IN, READOUT_TYPES,
    GNN_NODE_FEAT_OUT, AGGREGATORS,
    SCALERS
)
from ..layers.mlp import MLP
from .. import MultiLayerGNN


class HACTModel(BaseModel):
    """
    HACT model. The information for grading tumors lies at different scales. By building 2 graphs,
    one at the cell level and one at the object level (modeled with super pixels), we can extract graph embeddings
    that once combined provide a multi-scale representation of a RoI.

    This implementation is using GNN layers and spatial assignment matrix to fuse the 2 layers.
    """

    def __init__(
        self,
        cg_gnn_params: Dict,
        tg_gnn_params: Dict,
        classification_params: Dict,
        cg_node_dim: int,
        tg_node_dim: int,
        **kwargs
        ) -> None:
        """
        TissueGraphModel model constructor

        Args:
            cg_gnn_params: (dict) Cell Graph GNN configuration parameters.
            tg_gnn_params: (dict) Tissue Graph GNN configuration parameters.
            classification_params: (dict) classification configuration parameters.
            cg_node_dim (int): Cell node feature dimension. 
            tg_node_dim (int): Tissue node feature dimension. 
        """

        super().__init__(**kwargs)

        # 1- set class attributes
        self.cg_node_dim = cg_node_dim
        self.tg_node_dim = tg_node_dim
        self.cg_gnn_params = cg_gnn_params
        self.tg_gnn_params = tg_gnn_params
        self.classification_params = classification_params
        self.readout_op = cg_gnn_params['readout_op']
        self.with_rlp = False

        assert self.readout_op == tg_gnn_params['readout_op'], "Please the same readout operator for TG and CG. Options are 'concat', 'lstm', 'none'"

        # 2- build cell graph params
        self._build_cell_graph_params()

        # 3- build super pixel graph params
        if self.readout_op == "concat":
            self.cg_tg_node_dim = self.tg_node_dim +\
                self.cg_gnn_params['output_dim'] * self.cg_gnn_params['num_layers']
        else:
            self.cg_tg_node_dim = self.tg_node_dim + \
                self.cg_gnn_params['output_dim']

        self._build_tissue_graph_params()

        # 4- build classification params
        self._build_classification_params()

    def _build_tissue_graph_params(self):
        """
        Build multi layer GNN for tissue processing.
        """
        self.superpx_gnn = MultiLayerGNN(
            input_dim=self.cg_tg_node_dim,
            **self.tg_gnn_params
        )

    def _build_cell_graph_params(self):
        """
        Build cell graph multi layer GNN
        """
        self.cell_graph_gnn = MultiLayerGNN(
            input_dim=self.cg_node_dim,
            **self.cg_gnn_params
            )

    def _build_classification_params(self):
        """
        Build classification parameters
        """
        if self.readout_op == "concat":
            emd_dim = self.tg_gnn_params['output_dim'] * self.tg_gnn_params['num_layers']
        else:
            emd_dim = self.tg_gnn_params['output_dim']

        self.pred_layer = MLP(
            in_dim=emd_dim,
            h_dim=self.classification_params['hidden_dim'],
            out_dim=self.num_classes,
            num_layers=self.classification_params['num_layers']
        )

    def _compute_assigned_feats(self, graph, feats, assignment):
        """
        Use the assignment matrix to agg the feats

        Args:
            graph: (DGLBatch)
            feats: (FloatTensor)
            assignment: (list of LongTensor)
        """
        num_nodes_per_graph = graph.batch_num_nodes
        num_nodes_per_graph.insert(0, 0)
        intervals = [sum(num_nodes_per_graph[:i + 1])
                     for i in range(len(num_nodes_per_graph))]

        ll_h_concat = []
        for i in range(1, len(intervals)):
            h_agg = torch.matmul(
                assignment[i - 1], feats[intervals[i - 1]:intervals[i], :]
            )
            ll_h_concat.append(h_agg)

        return torch.cat(ll_h_concat, dim=0)

    def forward(
        self,
        cell_graph: Union[dgl.DGLGraph, dgl.batch],
        tissue_graph: Union[dgl.DGLGraph, dgl.batch],
        assignment_matrix: torch.Tensor
        ) -> torch.Tensor:
        """
        Foward pass.
        
        Args:
            cell_graph (Union[dgl.DGLGraph, dgl.batch]): Cell graph or Batch of cell graphs. 
            tissue_graph (Union[dgl.DGLGraph, dgl.batch]): Tissue graph or Batch of tissue graphs. 
            assignment_matrix (torch.Tensor): List of assignment matrices
        """

        # 1. GNN layers over the low level graph
        ll_feats = cell_graph.ndata[GNN_NODE_FEAT_IN]
        ll_h = self.cell_graph_gnn(cell_graph, ll_feats, with_readout=False)

        # 2. Sum the low level features according to assignment matrix
        ll_h_concat = self._compute_assigned_feats(
            cell_graph, ll_h, assignment_matrix)

        tissue_graph.ndata[GNN_NODE_FEAT_IN] = torch.cat(
            (ll_h_concat, tissue_graph.ndata[GNN_NODE_FEAT_IN]), dim=1)

        # 3. GNN layers over the high level graph
        hl_feats = tissue_graph.ndata[GNN_NODE_FEAT_IN]
        graph_embeddings = self.superpx_gnn(tissue_graph, hl_feats)

        # 4. Classification layers
        logits = self.pred_layer(graph_embeddings)
        return logits

    def set_rlp(self, with_rlp):
        raise NotImplementedError('LRP not implemented for HACT model.')

    def rlp(self, out_relevance_score):
        raise NotImplementedError('LRP not implemented for HACT model.')
