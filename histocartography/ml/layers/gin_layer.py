"""
Implementation of a GIN (Graph Isomorphism Network) layer.
In the  implementation the edges can also have weights that can be set as g.edata[GNN_EDGE_WEIGHT] = weight.

Original paper:
    - How Powerful are Graph Neural Networks: https://arxiv.org/abs/1810.00826
    - Author's public implementation: https://github.com/weihua916/powerful-gnns
"""

import torch
import torch.nn as nn

from histocartography.ml.layers.constants import GNN_MSG, GNN_NODE_FEAT_IN, GNN_NODE_FEAT_OUT, GNN_AGG_MSG, GNN_EDGE_WEIGHT
from histocartography.ml.layers.mlp import MLP


class GINLayer(nn.Module):

    def __init__(
            self,
            node_dim,
            hidden_dim,
            out_dim,
            act,
            layer_id,
            use_bn=True,
            config=None,
            verbose=False):
        """
        GIN Layer constructor
        :param node_dim: (int) input dimension of each node
        :param hidden_dim: (int) hidden dimension of each node (2-layer MLP as update function)
        :param out_dim: (int) output dimension of each node
        :param act: (str) activation function of the update function
        :param layer_id: (int) layer number
        :param use_bn: (bool) if layer uses batch norm
        :param config: (dict) optional argument
        :param verbose: (bool) verbosity level
        """
        super(GINLayer, self).__init__()

        if verbose:
            print('Creating new GNN layer:')

        if config is not None:
            eps = config['eps'] if 'eps' in config.keys() else None
            neighbor_pooling_type = config['neighbor_pooling_type'] if 'neighbor_pooling_type' in config.keys() else 'sum'
            learn_eps = config['learn_eps'] if 'learn_eps' in config.keys() else None
        else:
            eps = None
            neighbor_pooling_type = 'sum'
            learn_eps = None

        self.mlp = MLP(
            node_dim,
            hidden_dim,
            out_dim,
            2,
            act,
            use_bn,
            verbose=verbose)
        self._set_neighbor_pooling(neighbor_pooling_type)
        self.eps = eps
        self.learn_eps = learn_eps
        self.layer_id = layer_id

    def _set_neighbor_pooling(self, type):
        if type == 'sum':
            self.reduce_fn = self.sum_reduce_fn
        elif type == 'mean':
            self.reduce_fn = self.mean_reduce_fn
        else:
            raise ValueError(
                'Reduce type {} not implemented'.format(type))

    def sum_reduce_fn(self, nodes):
        """
        For each node, computes the sum of the neighbors
        """
        accum = torch.sum((nodes.mailbox[GNN_MSG]), dim=1)
        return {GNN_AGG_MSG: accum}

    def mean_reduce_fn(self, nodes):
        """
        For each node, computes the mean of the neighbors
        """
        accum = torch.mean(nodes.mailbox[GNN_MSG], dim=1)
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
        else:
            msg = edges.src[GNN_NODE_FEAT_IN]
        return {GNN_MSG: msg}

    def node_update_fn(self, nodes):
        """
        Node update function
        """
        h = nodes.data[GNN_NODE_FEAT_OUT]
        h = self.mlp(h)
        return {GNN_NODE_FEAT_OUT: h}

    def forward(self, g, h):
        """
        Forward-pass of a GIN layer.
        :param g: DGLGraph object. Node features in GNN_NODE_FEAT_IN_KEY
        :return: updated node features
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

        return g.ndata.pop(GNN_NODE_FEAT_OUT)
