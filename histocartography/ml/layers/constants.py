from torch.nn import ReLU, Tanh, Sigmoid, ELU, LeakyReLU

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
GNN_LL_NODE_FEAT = 'gnn_ll_node_feat'

AVAILABLE_LAYER_TYPES = {
    'gin_layer': 'GINLayer',
    'diff_pool_layer': 'DiffPoolLayer',
    'dense_gin_layer': 'DenseGINLayer'
}

GNN_MODULE = 'histocartography.ml.layers.{}'

