"""Torch modules for graph attention networks(GAT)."""
import torch as th
from torch import nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity

from histocartography.ml.layers.base_layer import BaseLayer
from histocartography.ml.layers.constants import ACTIVATIONS


class GATLayer(BaseLayer):
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
                 config=None,
                 act=None):

        super(GATLayer, self).__init__(node_dim, out_dim, act, layer_id)

        if config is not None:
            feat_drop = config['feat_drop']
            attn_drop = config['attn_drop']
            negative_slope = config['negative_slope']
            residual = config['residual']
            num_heads = config['num_heads']
        else:
            feat_drop = 0.
            attn_drop = 0.
            negative_slope = 0.2
            residual = False
            num_heads = 2

        out_dim = int(out_dim / num_heads)

        self._num_heads = num_heads
        self._in_feats = node_dim
        self._out_feats = out_dim
        self.fc = nn.Linear(node_dim, out_dim * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_dim)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_dim)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if residual:
            if node_dim != out_dim:
                self.res_fc = nn.Linear(node_dim, num_heads * out_dim, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = ACTIVATIONS[act]

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat):
        r"""Compute graph attention network layer.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """

        graph = graph.local_var()
        h = self.feat_drop(feat)
        feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
        el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.ndata.update({'ft': feat, 'el': el, 'er': er})

        # compute edge attention
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))

        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.ndata['ft']

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h).view(h.shape[0], -1, self._out_feats)
            rst = rst + resval

        # activation
        if self.activation:
            rst = self.activation(rst)

        return rst.flatten(1)
