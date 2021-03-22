"""
Implementation of a GIN (Graph Isomorphism Network) layer.

Original paper:
    - How Powerful are Graph Neural Networks: https://arxiv.org/abs/1810.00826
    - Author's public implementation: https://github.com/weihua916/powerful-gnns
"""
import itertools
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F

from histocartography.ml.layers.constants import (
    GNN_MSG, GNN_NODE_FEAT_IN, GNN_NODE_FEAT_OUT,
    GNN_AGG_MSG, GNN_EDGE_WEIGHT, REDUCE_TYPES,
    GNN_EDGE_FEAT
)
from .mlp import MLP
from .base_layer import BaseLayer


class GINLayer(BaseLayer):

    def __init__(
            self,
            node_dim: int,
            out_dim: int,
            act: str = 'relu',
            agg_type: str = 'sum',
            hidden_dim: int = 32,
            use_bn: bool = True,
            graph_norm: bool = False,
            with_lrp: bool = False,
            dropout: float = 0.,
            verbose: bool = False) -> None:
        """
        GIN Layer constructor

        Args:
            node_dim (int): input dimension of each node
            out_dim (int): output dimension of each node
            act (str): activation function of the update function
            use_bn (bool): if layer uses batch norm
            agg_type (str): Aggreagtion function. Default to 'sum'.
            hidden_dim (int): Hidden dimension of the GIN MLP. Default to 32.
            use_bn (bool): If we should use batch normalization. Default to True.
            graph_norm (bool): If we should use graph normalization. Default to False.
            with_lrp (bool): If we should use LRP. Default to False.
            dropout (float): If we should use dropout. Default to 0.
            verbose (bool): Verbosity. Default to False. 
        """
        super().__init__(node_dim, out_dim, act)

        if verbose:
            print('Instantiating new GNN layer.')

        self.use_bn = use_bn 
        self.graph_norm = graph_norm
        self.with_lrp = with_lrp

        if self.use_bn:
            self.batchnorm_h = nn.BatchNorm1d(out_dim)

        self.mlp = MLP(
            node_dim,
            hidden_dim,
            out_dim,
            2,
            act,
            verbose=verbose,
            dropout=dropout,
            with_lrp=self.with_lrp
        )

        self.agg_type = agg_type

    def reduce_fn(self, nodes):
        """
        For each node, aggregate the nodes using a reduce function.
        Current supported functions are sum and mean.
        """
        accum = REDUCE_TYPES[self.agg_type](
            (nodes.mailbox[GNN_MSG]), dim=1)
        return {GNN_AGG_MSG: accum}

    def msg_fn(self, edges):
        """
        Message of each node
        """
        msg = edges.src[GNN_NODE_FEAT_IN]
        return {GNN_MSG: msg}

    def node_update_fn(self, nodes):
        """
        Node update function
        """
        h = nodes.data[GNN_NODE_FEAT_IN]
        h = self.mlp(h)
        h = F.relu(h)
        return {GNN_NODE_FEAT_OUT: h}

    def forward(self, g, h):
        """
        Forward-pass of a GIN layer.
        :param g: (DGLGraph) graph to process.
        :param h: (FloatTensor) node features
        """

        if self.with_lrp:
            self.adjacency_matrix = g.adjacency_matrix(ctx=h.device).to_dense()
            self.input_features = h.t()

        g.ndata[GNN_NODE_FEAT_IN] = h

        g.update_all(self.msg_fn, self.reduce_fn)

        if GNN_AGG_MSG in g.ndata.keys():
            g.ndata[GNN_NODE_FEAT_IN] = g.ndata[GNN_AGG_MSG] + g.ndata[GNN_NODE_FEAT_IN]
        else:
            g.ndata[GNN_NODE_FEAT_IN] = g.ndata[GNN_NODE_FEAT_IN]

        g.apply_nodes(func=self.node_update_fn)

        # apply graph norm and batch norm 
        h = g.ndata[GNN_NODE_FEAT_OUT]
        if self.graph_norm:
            snorm_n = torch.FloatTensor(list(itertools.chain(*[[np.sqrt(1/n)] * n for n in g.batch_num_nodes]))).to(h.get_device())
            h = h * snorm_n[:, None]
        if self.use_bn:
            h = self.batchnorm_h(h)

        return h

    def set_lrp(self, with_lrp):
        self.with_lrp = with_lrp
        self.mlp.set_lrp(with_lrp)

    def _compute_adj_lrp(self, relevance_score):
        adjacency_matrix = self.adjacency_matrix + torch.eye(self.adjacency_matrix.shape[0]).to(relevance_score.device)
        V = torch.clamp(adjacency_matrix, min=0)  # @TODO: rename variables in this class. 
        Z = torch.mm(self.input_features, V.t()) + 1e-9
        S = relevance_score / Z.t()
        C = torch.mm(V, S)
        relevance_score = self.input_features.t() * C
        return relevance_score

    def lrp(self, out_relevance_score):
        """
        Implement lrp for GIN layer: 
        """

        # 1/ lrp over the node update function 
        relevance_score = self.mlp.lrp(out_relevance_score)

        # 2/ lrp over the adjacency 
        relevance_score = self._compute_adj_lrp(relevance_score)

        return relevance_score
