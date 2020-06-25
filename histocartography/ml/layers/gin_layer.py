"""
Implementation of a GIN (Graph Isomorphism Network) layer.
In the  implementation the edges can also have weights that can be set as g.edata[GNN_EDGE_WEIGHT] = weight.

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
from histocartography.ml.layers.mlp import MLP
from histocartography.ml.layers.base_layer import BaseLayer


class GINLayer(BaseLayer):

    def __init__(
            self,
            node_dim,
            out_dim,
            act,
            layer_id,
            edge_dim=None,
            config=None,
            verbose=False):
        """
        GIN Layer constructor
        :param node_dim: (int) input dimension of each node
        :param out_dim: (int) output dimension of each node
        :param act: (str) activation function of the update function
        :param layer_id: (int) layer number
        :param use_bn: (bool) if layer uses batch norm
        :param config: (dict) optional argument
        :param verbose: (bool) verbosity level
        """
        super(
            GINLayer,
            self).__init__(
            node_dim,
            out_dim,
            act,
            layer_id)

        if verbose:
            print('Creating new GNN layer:')

        if config is not None:
            eps = config['eps'] if 'eps' in config.keys() else None
            neighbor_pooling_type = config['neighbor_pooling_type'] if 'neighbor_pooling_type' in config.keys(
            ) else 'sum'
            learn_eps = config['learn_eps'] if 'learn_eps' in config.keys(
            ) else None
            hidden_dim = config['hidden_dim']
            self.use_bn = config['use_bn'] if 'use_bn' in config.keys() else True
            dropout = config['dropout'] if 'dropout' in config.keys() else 0.
            self.graph_norm = config['graph_norm'] if 'graph_norm' in config.keys() else False
        else:
            eps = None
            neighbor_pooling_type = 'sum'
            learn_eps = None
            hidden_dim = 32
            self.use_bn = True
            dropout = 0.
            self.graph_norm = False 

        if self.use_bn:
            self.batchnorm_h = nn.BatchNorm1d(out_dim)

        if edge_dim is not None:
            edge_encoding_dim = min(node_dim, 16)  # hardcoded param setting the edge encoding to 16
            self.edge_encoder = nn.Linear(edge_dim, edge_encoding_dim)

        print('Config in GIN layer:', config)

        self.mlp = MLP(
            node_dim,
            hidden_dim,
            out_dim,
            2,
            act,
            use_bn=False,  # bn is put after in the node update fn 
            verbose=verbose,
            dropout=dropout
        )

        self.eps = eps
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.layer_id = layer_id

    def reduce_fn(self, nodes):
        """
        For each node, aggregate the nodes using a reduce function.
        Current supported functions are sum and mean.
        """
        accum = REDUCE_TYPES[self.neighbor_pooling_type](
            (nodes.mailbox[GNN_MSG]), dim=1)
        return {GNN_AGG_MSG: accum}

    def msg_fn(self, edges):
        """
        Message of each node
        """
        if GNN_EDGE_WEIGHT in edges.data.keys():
            msg = edges.src[GNN_NODE_FEAT_IN] * torch.t(
                edges.data[GNN_EDGE_WEIGHT].repeat(
                    edges.src[GNN_NODE_FEAT_IN].shape[1],
                    1))
        elif GNN_EDGE_FEAT in edges.data.keys():
            efeat = self.edge_encoder(edges.data[GNN_EDGE_FEAT])
            padded_efeat = torch.zeros(edges.src[GNN_NODE_FEAT_IN].shape).to(edges.src[GNN_NODE_FEAT_IN].device)
            padded_efeat[:, :efeat.shape[1]] = efeat
            msg = edges.src[GNN_NODE_FEAT_IN] + padded_efeat
        else:
            msg = edges.src[GNN_NODE_FEAT_IN]
        return {GNN_MSG: msg}

    def node_update_fn(self, nodes):
        """
        Node update function
        """
        h = nodes.data[GNN_NODE_FEAT_OUT]
        h = self.mlp(h)
        h = F.relu(h)
        return {GNN_NODE_FEAT_OUT: h}

    def forward(self, g, h):
        """
        Forward-pass of a GIN layer.
        :param g: (DGLGraph) graph to process.
        :param h: (FloatTensor) node features
        """
        g.ndata[GNN_NODE_FEAT_IN] = h

        g.update_all(self.msg_fn, self.reduce_fn)

        if self.learn_eps:
            g.ndata[GNN_NODE_FEAT_OUT] = g.ndata[GNN_AGG_MSG] + \
                (1 + self.eps[self.layer_id]) * g.ndata[GNN_NODE_FEAT_IN]
        else:
            g.ndata[GNN_NODE_FEAT_OUT] = g.ndata[GNN_AGG_MSG] + \
                g.ndata[GNN_NODE_FEAT_IN]

        g.apply_nodes(func=self.node_update_fn)

        # apply graph norm and batch norm 
        h = g.ndata[GNN_NODE_FEAT_OUT]
        if self.graph_norm:
            snorm_n = torch.FloatTensor(list(itertools.chain(*[[np.sqrt(1/n)] * n for n in g.batch_num_nodes]))).to(h.get_device())
            h = h * snorm_n[:, None]
        if self.use_bn:
            h = self.batchnorm_h(h)

        return h
