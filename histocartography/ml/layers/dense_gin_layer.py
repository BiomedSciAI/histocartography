"""
Implementation of a Dense GIN (Graph Isomorphism Network) layer. This implementation should be used
when the input graph(s) can only be represented as an adjacency (typically when dealing with dense
adjacency matrices).

Original paper:
    - How Powerful are Graph Neural Networks: https://arxiv.org/abs/1810.00826
    - Author's public implementation: https://github.com/weihua916/powerful-gnns
"""

import torch
import torch.nn.functional as F

from histocartography.ml.layers.mlp import MLP
from histocartography.ml.layers.base_layer import BaseLayer


class DenseGINLayer(BaseLayer):

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
        :param add_self: (bool) add self loops to the input adjacency
        :param mean: (bool) adjust the adjacency with its mean
        :param verbose: (bool) verbosity level
        """
        super(DenseGINLayer, self).__init__(node_dim, hidden_dim, out_dim, act, layer_id)

        if verbose:
            print('Creating new GNN layer:')

        if config is not None:
            self.add_self = config['add_self'] if 'add_self' in config.keys() else True
            self.mean = config['mean'] if 'mean' in config.keys() else False
        else:
            self.add_self = True
            self.mean = False

        self.mlp = MLP(
            node_dim,
            hidden_dim,
            out_dim,
            2,
            act,
            use_bn,
            verbose=verbose)
        self.layer_id = layer_id

    def forward(self, adj, h, cat=False):
        """
        Forward-pass of a Dense GIN layer.
        :param g: DGLGraph object. Node features in GNN_NODE_FEAT_IN_KEY
        :return: updated node features
        """
        # @TODO implement cat operator.

        if self.add_self:
            adj = adj + torch.eye(adj.size(1)).to(adj.device)

        if self.mean:
            adj = adj / adj.sum(1, keepdim=True)

        h_k_N = torch.matmul(adj, h)
        bs, n_nodes, dim = h_k_N.shape
        h_k_N = h_k_N.view(bs * n_nodes, dim)

        h_k = self.mlp(h_k_N)
        h_k = h_k.view(bs, n_nodes, dim)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        return h_k
