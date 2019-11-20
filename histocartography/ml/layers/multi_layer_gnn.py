import torch.nn as nn
import importlib

from histocartography.ml.layers.constants import AVAILABLE_LAYER_TYPES, GNN_MODULE


class MultiLayerGNN(nn.Module):
    """
    MultiLayer network that concatenate several gnn layers layer
    """

    def __init__(self, config):
        super(MultiLayerGNN, self).__init__()

        layer_type = config['layer_type']
        if layer_type in list(AVAILABLE_LAYER_TYPES.keys()):
            module = importlib.import_module(
                GNN_MODULE.format(layer_type)
            )
        else:
            raise ValueError(
                'GNN type: {} not recognized. Options are: {}'.format(
                    layer_type, list(AVAILABLE_LAYER_TYPES.keys())
                )
            )

        in_dim = config['in_dim']
        hidden_dim = config['hidden_dim']
        out_dim = config['out_dim']
        num_layers = config['num_layers']
        activation = config['activation']
        use_bn = config['use_bn']
        neighbor_pooling_type = config['neighbor_pooling_type']

        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(getattr(module, AVAILABLE_LAYER_TYPES[layer_type])(
            node_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            act=activation,
            layer_id=0,
            use_bn=use_bn,
            neighbor_pooling_type=neighbor_pooling_type)
        )
        # hidden layers
        for i in range(1, num_layers-1):
            self.layers.append(getattr(module, AVAILABLE_LAYER_TYPES[layer_type])(
                node_dim=hidden_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                act=activation,
                layer_id=i,
                use_bn=use_bn,
                neighbor_pooling_type=neighbor_pooling_type)
            )
        # output layer
        self.layers.append(getattr(module, AVAILABLE_LAYER_TYPES[layer_type])(
            node_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            act=activation,
            layer_id=num_layers-1,
            use_bn=use_bn,
            neighbor_pooling_type=neighbor_pooling_type)
        )

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h
