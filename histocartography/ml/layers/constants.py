import torch
from torch.nn import ReLU, Tanh, Sigmoid, ELU, LeakyReLU
import dgl


ACTIVATIONS = {
    'relu': ReLU(),
    'tanh': Tanh(),
    'sigmoid': Sigmoid(),
    'elu': ELU(),
    'leaky_relu': LeakyReLU()
}

GNN_MSG = 'gnn_msg'
GNN_NODE_FEAT_IN = 'feat'
GNN_NODE_FEAT_OUT = 'gnn_node_feat_out'
GNN_AGG_MSG = 'gnn_agg_msg'
GNN_EDGE_WEIGHT = 'gnn_edge_weight'

AVAILABLE_LAYER_TYPES = {
    'gin_layer': 'GINLayer',
    'diff_pool_layer': 'DiffPoolLayer',
    'dense_gin_layer': 'DenseGINLayer',
    'pooled_gin_layer': 'PooledGINLayer'
}

GNN_MODULE = 'histocartography.ml.layers.{}'

READOUT_TYPES = {
    'sum': dgl.sum_nodes,
    'mean': dgl.mean_nodes,
    'max': dgl.max_nodes
}

REDUCE_TYPES = {
    'sum': torch.sum,
    'mean': torch.mean
}
