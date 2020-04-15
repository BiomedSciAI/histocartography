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
import dgl

from histocartography.ml.layers.mlp import MLP
from histocartography.ml.layers.base_layer import BaseLayer


class DenseGINLayer(BaseLayer):

    def __init__(
            self,
            node_dim,
            out_dim,
            act,
            layer_id,
            use_bn=False,
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
        super(
            DenseGINLayer,
            self).__init__(
            node_dim,
            out_dim,
            act,
            layer_id)

        if verbose:
            print('Creating new GNN dense layer:')

        self.out_dim = out_dim

        if config is not None:
            self.add_self = True
            self.mean = config['neighbor_pooling_type'] == 'mean'
            hidden_dim = config['hidden_dim']
        else:
            self.add_self = True
            self.mean = False
            hidden_dim = 32

        self.mlp = MLP(
            node_dim,
            hidden_dim,
            out_dim,
            2,
            act,
            use_bn,
            verbose=verbose)
        self.layer_id = layer_id

    def forward(self, adj, h):
        """
        Forward-pass of a Dense GIN layer.
        :param g: DGLGraph object. Node features in GNN_NODE_FEAT_IN_KEY
        :return: updated node features
        """

        if isinstance(adj, dgl.DGLGraph):
            adj = dgl.unbatch(adj)
            assert(len(adj) == 1), "Batch size must be equal to 1 for processing Dense GIN Layers"
            adj = adj[0].adjacency_matrix().to_dense().unsqueeze(dim=0).to(h.device)

        if self.mean:
            degree = adj.sum(1, keepdim=True)
            degree[degree == 0.] = 1.
            adj = adj / degree

        if self.add_self:
            adj = adj + torch.eye(adj.size(1)).to(adj.device)

        # adjust h dim
        if len(h.shape) < 3:
            h = h.unsqueeze(dim=0)

        h_k_N = torch.matmul(adj, h)
        bs, n_nodes, dim = h_k_N.shape
        h_k_N = h_k_N.view(bs * n_nodes, dim)
        h_k = self.mlp(h_k_N)
        h_k = h_k.view(bs, n_nodes, self.out_dim)
        h_k = F.relu(h_k).squeeze()
        return h_k
