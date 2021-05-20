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
from torch import nn

from .mlp import MLP


class DenseGINLayer(nn.Module):

    def __init__(
            self,
            node_dim: int,
            out_dim: int,
            act: str = 'relu',
            agg_type: str = 'mean',
            hidden_dim: int = 32,
            batch_norm: bool = True,
            graph_norm: bool = False,
            with_lrp: bool = False,
            dropout: float = 0.,
            verbose: bool = False) -> None:
        """
        Dense GIN Layer constructor

        Args:
            node_dim (int): Input dimension of each node.
            out_dim (int): Output dimension of each node.
            act (str): Activation function of the update function.
            agg_type (str): Aggregation function. Defaults to 'mean'.
            hidden_dim (int): Hidden dimension of the GIN MLP. Defaults to 32.
            batch_norm (bool): If we should use batch normalization. Defaults to True.
            graph_norm (bool): If we should use graph normalization. Defaults to False.
            with_lrp (bool): If we should use LRP. Defaults to False.
            dropout (float): If we should use dropout. Defaults to 0.
            verbose (bool): Verbosity. Defaults to False.
        """

        super().__init__()

        if verbose:
            print('Creating dense GIN layer:')

        self.out_dim = out_dim
        self.add_self = True
        self.mean = agg_type == 'mean'

        self.mlp = MLP(
            node_dim,
            hidden_dim,
            out_dim,
            2,
            act,
            batch_norm,
            verbose=verbose)

    def forward(self, adj, h):
        """
        Forward-pass of a Dense GIN layer.
        :param g: DGLGraph object. Node features in GNN_NODE_FEAT_IN_KEY
        :return: updated node features
        """

        if isinstance(adj, dgl.DGLGraph):
            adj = dgl.unbatch(adj)
            assert(
                len(adj) == 1), "Batch size must be equal to 1 for processing Dense GIN Layers"
            adj = adj[0].adjacency_matrix().to_dense().unsqueeze(
                dim=0).to(
                h.device)

        if self.mean:
            degree = adj.sum(1, keepdim=True)
            degree[degree == 0.] = 1.
            adj = adj / degree

        if self.add_self:
            adj = adj.float() + torch.eye(adj.size(1)).to(adj.device)

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
