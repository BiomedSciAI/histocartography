import itertools
import math 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from histocartography.ml.layers.constants import AGGREGATORS, SCALERS, GNN_NODE_FEAT_IN, GNN_EDGE_FEAT
from histocartography.ml.layers.mlp import MLP


"""
    PNA: Principal Neighbourhood Aggregation 
    Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, Petar Velickovic
    https://arxiv.org/abs/2004.05718
"""

class PNATower(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, avg_d,
                 pretrans_layers, posttrans_layers, edge_dim):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.edge_dim = edge_dim

        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

        self.aggregators = aggregators
        self.scalers = scalers
        self.pretrans = nn.Sequential(
            MLP(in_dim=2 * in_dim + (edge_dim if edge_dim is not None else 0), h_dim=out_dim, out_dim=out_dim, num_layers=pretrans_layers, act='relu'),
            nn.ReLU()
        )
        self.posttrans = nn.Sequential(
            MLP(in_dim=(len(aggregators) * len(scalers)) * out_dim + in_dim, h_dim=out_dim, out_dim=out_dim, num_layers=posttrans_layers, act='relu'),
            nn.ReLU()
        )
        self.avg_d = avg_d

    def pretrans_edges(self, edges):
        if self.edge_dim is not None:
            z2 = torch.cat([edges.src[GNN_NODE_FEAT_IN], edges.dst[GNN_NODE_FEAT_IN], edges.data[GNN_EDGE_FEAT]], dim=1)
        else:
            z2 = torch.cat([edges.src[GNN_NODE_FEAT_IN], edges.dst[GNN_NODE_FEAT_IN]], dim=1)
        return {'e': self.pretrans(z2)}

    def message_func(self, edges):
        return {'e': edges.data['e']}

    def reduce_func(self, nodes):
        h = nodes.mailbox['e']
        D = h.shape[-2]
        h = torch.cat([aggregate(h) for aggregate in self.aggregators], dim=1)
        h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)
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
            snorm_n = torch.FloatTensor(list(itertools.chain(*[[np.sqrt(1/n)] * n for n in g.batch_num_nodes]))).to(h.get_device())
            h = h * snorm_n[:, None]
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class PNALayer(nn.Module):

    def __init__(self,
                 node_dim,
                 out_dim,
                 act,
                 layer_id,
                 config=None,
                 edge_dim=None,
                 verbose=False
                 ):
        """
        :param node_dim:              size of the input per node
        :param out_dim:             size of the output per node
        :param aggregators:         set of aggregation function identifiers
        :param scalers:             set of scaling functions identifiers
        :param avg_d:               average degree of nodes in the training set, used by scalers to normalize
        :param dropout:             dropout used
        :param graph_norm:          whether to use graph normalisation
        :param batch_norm:          whether to use batch normalisation
        :param towers:              number of towers to use
        :param pretrans_layers:     number of layers in the transformation before the aggregation
        :param posttrans_layers:    number of layers in the transformation after the aggregation
        :param divide_input:        whether the input features should be split between towers or not
        :param residual:            whether to add a residual connection
        """
        super().__init__()

        aggregators = config['aggregators']
        scalers = config['scalers']
        avg_d = {'log': math.log(4 + 1)}  # @TODO: hardcoded parameters
        dropout = config['dropout']
        graph_norm = config['graph_norm']
        batch_norm = config['use_bn']
        towers = config['towers']
        pretrans_layers = config['pretrans_layers']
        posttrans_layers = config['pretrans_layers']
        divide_input= config['divide_input_first'] 
        residual = config['residual']

        assert ((not divide_input) or node_dim % towers == 0), "if divide_input is set the number of towers has to divide node_dim"
        assert (out_dim % towers == 0), "the number of towers has to divide the out_dim"
        assert avg_d is not None

        # retrieve the aggregators and scalers functions
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators.split()]
        scalers = [SCALERS[scale] for scale in scalers.split()]

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
            self.towers.append(PNATower(in_dim=self.input_tower, out_dim=self.output_tower, aggregators=aggregators,
                                        scalers=scalers, avg_d=avg_d, pretrans_layers=pretrans_layers,
                                        posttrans_layers=posttrans_layers, batch_norm=batch_norm, dropout=dropout,
                                        graph_norm=graph_norm, edge_dim=edge_dim))
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


def __repr__(self):
    return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__, self.node_dim, self.out_dim)
