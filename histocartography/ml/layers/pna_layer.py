"""
    PNA: Principal Neighbourhood Aggregation
    Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, Petar Velickovic
    https://arxiv.org/abs/2004.05718
"""

import itertools
import math
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import AGGREGATORS, SCALERS, GNN_NODE_FEAT_IN, GNN_EDGE_FEAT
from .mlp import MLP


class PNALayer(nn.Module):

    def __init__(self,
                 node_dim,
                 out_dim,
                 aggregators: str = "mean max min std",
                 scalers: str = "identity amplification attenuation",
                 avg_d: int = 4,
                 dropout: float = 0.,
                 graph_norm: bool = False,
                 batch_norm: bool = False,
                 towers: int = 1,
                 pretrans_layers: int = 1,
                 posttrans_layers: int = 1,
                 divide_input: bool = True,
                 residual: bool = True,
                 verbose=False
                 ):
        """
        PNA layer constructor.

        Args:
            node_dim (int): Input dimension of each node.
            out_dim (int): Output dimension of each node.
            aggregators (str): Set of aggregation function identifiers. Default to "mean max min std".
            scalers (str): Set of scaling functions identifiers. Default to "identity amplification attenuation".
            avg_d (int): Average degree of nodes in the training set, used by scalers to normalize. Default to 5.
            dropout (float): Dropout used. Default to 0.
            graph_norm (bool): Whether to use graph normalisation. Default to False.
            batch_norm (bool): Whether to use batch normalisation. Default to False.
            towers: Number of towers to use. Default to 1.
            pretrans_layers: Number of layers in the transformation before the aggregation. Default to 1.
            posttrans_layers: Number of layers in the transformation after the aggregation. Default to 1.
            divide_input: Whether the input features should be split between towers or not. Default to True.
            residual: Whether to add a residual connection. Default to True.
            verbose (bool): Verbosity. Default to False.
        """
        super().__init__()

        if verbose:
            print('Instantiating new GNN layer.')

        assert ((not divide_input) or node_dim % towers ==
                0), "if divide_input is set the number of towers has to divide node_dim"
        assert (
            out_dim %
            towers == 0), "the number of towers has to divide the out_dim"

        # retrieve the aggregators, scalers functions and avg degree
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators.split()]
        scalers = [SCALERS[scale] for scale in scalers.split()]
        avg_d = {'log': math.log(avg_d + 1)}

        self.divide_input = divide_input
        self.input_tower = node_dim // towers if divide_input else node_dim
        self.output_tower = out_dim // towers
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.residual = residual
        if node_dim != out_dim:
            self.residual = False

        # convolution
        self.towers = nn.ModuleList()
        for _ in range(towers):
            self.towers.append(
                PNATower(
                    in_dim=self.input_tower,
                    out_dim=self.output_tower,
                    aggregators=aggregators,
                    scalers=scalers,
                    avg_d=avg_d,
                    pretrans_layers=pretrans_layers,
                    posttrans_layers=posttrans_layers,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    graph_norm=graph_norm))
        # mixing network
        self.mixing_network = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU()
        )

    def forward(self, g, h):
        h_in = h  # for residual connection

        if self.divide_input:
            h_cat = torch.cat(
                [tower(g, h[:, n_tower * self.input_tower: (n_tower + 1) * self.input_tower])
                 for n_tower, tower in enumerate(self.towers)], dim=1)
        else:
            h_cat = torch.cat([tower(g, h) for tower in self.towers], dim=1)

        h_out = self.mixing_network(h_cat)

        if self.residual:
            h_out = h_in + h_out  # residual connection
        return h_out

    def set_rlp(self, with_rlp):
        raise NotImplementedError(
            'LRP not implemented for PNA layers. Use a GIN-based model.')


def __repr__(self):
    return '{}(in_channels={}, out_channels={})'.format(
        self.__class__.__name__, self.node_dim, self.out_dim)


class PNATower(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            dropout,
            graph_norm,
            batch_norm,
            aggregators,
            scalers,
            avg_d,
            pretrans_layers,
            posttrans_layers):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm

        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

        self.aggregators = aggregators
        self.scalers = scalers
        self.pretrans = nn.Sequential(
            MLP(
                in_dim=2 * in_dim,
                hidden_dim=out_dim,
                out_dim=out_dim,
                num_layers=pretrans_layers,
                act='relu'),
            nn.ReLU())
        self.posttrans = nn.Sequential(
            MLP(
                in_dim=(
                    len(aggregators) *
                    len(scalers)) *
                out_dim +
                in_dim,
                hidden_dim=out_dim,
                out_dim=out_dim,
                num_layers=posttrans_layers,
                act='relu'),
            nn.ReLU())
        self.avg_d = avg_d

    def pretrans_edges(self, edges):
        z2 = torch.cat([edges.src[GNN_NODE_FEAT_IN],
                        edges.dst[GNN_NODE_FEAT_IN]], dim=1)
        return {'e': self.pretrans(z2)}

    def message_func(self, edges):
        return {'e': edges.data['e']}

    def reduce_func(self, nodes):
        h = nodes.mailbox['e']
        D = h.shape[-2]
        h = torch.cat([aggregate(h) for aggregate in self.aggregators], dim=1)
        h = torch.cat([scale(h, D=D, avg_d=self.avg_d)
                       for scale in self.scalers], dim=1)
        return {GNN_NODE_FEAT_IN: h}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data[GNN_NODE_FEAT_IN])

    def forward(self, g, h):
        g.ndata[GNN_NODE_FEAT_IN] = h

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata[GNN_NODE_FEAT_IN]], dim=1)

        # posttransformation
        h = self.posttrans(h)

        # graph and batch normalization
        if self.graph_norm:
            if hasattr(g, 'batch_num_nodes'):
                num_nodes = g.batch_num_nodes
            else:
                num_nodes = [g.number_of_nodes()]
            snorm_n = torch.FloatTensor(list(itertools.chain(*[[np.sqrt(1 / n)] * n for n in num_nodes]))).to(h.device)
            h = h * snorm_n[:, None]
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
