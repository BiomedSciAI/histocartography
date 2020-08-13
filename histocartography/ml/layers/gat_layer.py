"""Torch modules for graph attention networks(GAT)."""
from torch import nn
import torch
import dgl.function as fn
import torch.nn.functional as F

from histocartography.ml.layers.base_layer import BaseLayer
from histocartography.ml.layers.constants import ACTIVATIONS
from histocartography.ml.layers.constants import (
    GNN_MSG, GNN_NODE_FEAT_IN, GNN_NODE_FEAT_OUT,
    GNN_AGG_MSG, GNN_EDGE_WEIGHT, REDUCE_TYPES,
    GNN_EDGE_FEAT
)


class SingleHeadGATLayer(BaseLayer):
    r"""Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    over an input signal.
    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}
    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:
    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})
        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)
    Parameters:
    ----------
    :param node_dim: (int) Input feature size
    :param out_dim: (int) Output feature size
    :param layer_id: (int) Layer id
    :param act: (str) callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    :param config : dict
    """

    def __init__(self,
                 node_dim,
                 out_dim,
                 layer_id,
                 edge_dim=None,
                 config=None,
                 act=None):

        super(SingleHeadGATLayer, self).__init__(node_dim, out_dim, act, layer_id)

        # equation (1)
        self.fc = nn.Linear(node_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.attn_weights = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        a = F.leaky_relu(a)

        # update attention weights for further analysis 
        self.attn_weights = a

        return {'e': a}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {GNN_NODE_FEAT_OUT: h}

    def forward(self, graph, feats):
        # equation (1)
        z = self.fc(feats)
        graph.ndata['z'] = z
        # equation (2)
        graph.apply_edges(self.edge_attention)
        # equation (3) & (4)
        graph.update_all(self.message_func, self.reduce_func)
        h = graph.ndata.pop(GNN_NODE_FEAT_OUT) # + z  # add self connections 
        return h 


class GATLayer(BaseLayer):
    def __init__(self,
                 node_dim,
                 out_dim,
                 layer_id,
                 edge_dim=None,
                 config=None,
                 act=None):

        super(GATLayer, self).__init__(node_dim, out_dim, act, layer_id)

        num_heads = config['num_heads']
        self.merge = config['merge']

        if self.merge == 'cat':
            out_dim = int(out_dim / num_heads)

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(SingleHeadGATLayer(node_dim, out_dim, i, config=config))

    def forward(self, graph, feats):
        head_outs = [attn_head(graph, feats) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        elif self.merge == 'avg':
            # merge using average
            return torch.mean(torch.stack(head_outs), dim=0)
        else:
            raise ValueError('Unsupported merge type in GAT layer. Options are "cat" and "avg".')

