""" The edGNN layer based on:
    "edGNN: A simple and efficient GNN for directed labeled graph"
"""

# Torch packages
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F

# Constants
from histocartography.ml.layers.constants import (
    GNN_MSG, GNN_NODE_FEAT_IN,
    GNN_NODE_FEAT_OUT, GNN_AGG_MSG,
    GNN_EDGE_FEAT, REDUCE_TYPES
)
from histocartography.ml.layers.mlp import MLP
from histocartography.ml.layers.base_layer import BaseLayer


class edGNNLayer(BaseLayer):
    def __init__(self,
                node_dim,
                out_dim,
                act,
                layer_id,
                config=None,
                verbose=False):
        super(edGNNLayer, self).__init__(node_dim, out_dim, act, layer_id)

        if config is not None:
            self.edge_dim = config['edge_dim'] if 'edge_dim' in config.keys() else None 
            self.neighbor_pooling_type = config['neighbor_pooling_type'] if 'neighbor_pooling_type' in config.keys(
            ) else 'sum'
            hidden_dim = config['hidden_dim']
            use_bn = config['use_bn'] if 'use_bn' in config.keys() else True
            dropout = config['dropout'] if 'dropout' in config.keys() else 0.
        else:
            self.neighbor_pooling_type = 'sum'
            hidden_dim = 32
            use_bn = True
            dropout = 0.

        self.mlp_update = MLP(
            node_dim,
            hidden_dim,
            out_dim,
            2,
            act,
            use_bn,
            verbose=verbose,
            dropout=dropout
        )

        self.mlp_aggreg = MLP(
            node_dim + self.edge_dim if self.edge_dim is not None else node_dim,
            hidden_dim,
            out_dim,
            2,
            act,
            use_bn,
            verbose=verbose,
            dropout=dropout
        )

    def gnn_msg(self, edges):
        """
            Include edge features: for each edge u->v, return as msg: MLP(concat([h_u, h_uv]))
        """
        msg = torch.cat([edges.src[GNN_NODE_FEAT_IN],
                            edges.data[GNN_EDGE_FEAT]],
                        dim=1)
        msg = self.mlp_aggreg(msg)
        return {GNN_MSG: msg}

    def gnn_reduce(self, nodes):
        accum = REDUCE_TYPES[self.neighbor_pooling_type]((nodes.mailbox[GNN_MSG]), 1)
        return {GNN_AGG_MSG: accum}

    def node_update(self, nodes):
        h = nodes.data[GNN_NODE_FEAT_IN]
        h = self.mlp_update(h)
        h = h + nodes.data.pop(GNN_AGG_MSG)
        h =  F.relu(h)
        return {GNN_NODE_FEAT_OUT: h}

    def forward(self, g, features):
        
        # 0. to double check 
        node_features, edge_features = features

        # 1. set current iteration features
        g.ndata[GNN_NODE_FEAT_IN] = node_features
        g.edata[GNN_EDGE_FEAT] = edge_features

        # 2. aggregate messages
        g.update_all(self.gnn_msg, self.gnn_reduce)
        g.apply_nodes(self.node_update)

        return g.ndata.pop(GNN_NODE_FEAT_OUT)
