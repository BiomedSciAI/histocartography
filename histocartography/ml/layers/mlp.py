import torch.nn as nn
from torch.nn import Sequential, Linear
import torch

from .constants import ACTIVATIONS


class MLP(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=1,
        act="relu",
        use_bn=False,
        bias=True,
        verbose=False,
        dropout=0.,
        with_lrp=False
    ):
        """
        MLP Constructor
        :param in_dim: (int) input dimension
        :param hidden_dim: (int) hidden dimension(s), if type(h_dim) is int => all the hidden have the same dimensions
                                                 if type(h_dim) is list => hidden use val in list as dimension
        :param out_dim: (int) output_dimension
        :param num_layers: (int) number of layers
        :param act: (str) activation function to use, last layer without activation!
        :param use_bn: (bool) il layers should have batch norm
        :param bias: is Linear should have bias term, if type(h_dim) is bool => all the hidden have bias terms
                                                      if type(h_dim) is list of bool => hidden use val in list as bias
        :param verbose: (bool) verbosity level
        """
        super(MLP, self).__init__()

        # optional argument
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # set activations
        self._set_activations(act)

        # set mlp dimensions
        self._set_mlp_dimensions(in_dim, hidden_dim, out_dim, num_layers)

        # set batch norm
        self._set_batch_norm(use_bn, num_layers)

        # set dropout
        self._set_dropout(dropout, num_layers)

        # set bias terms
        self._set_biases(bias, num_layers)

        # set lrp
        self.with_lrp = with_lrp
        if self.with_lrp:
            self.forward_activations = []

        # build MLP layers
        self.mlp = nn.ModuleList()
        if num_layers == 1:
            self.mlp = self._build_layer(0, act=False)
        elif num_layers > 1:
            # append hidden layers
            for layer_id in range(num_layers):
                self.mlp.append(
                    self._build_layer(
                        layer_id, act=layer_id != (
                            num_layers - 1)))
        else:
            raise ValueError('The number of layers must be greater than 1.')

        if verbose:
            for layer_id, layer in enumerate(self.mlp):
                print('MLP layer {} has params {}'.format(layer_id, layer))

    def _build_layer(self, layer_id, act=True):
        """
        Build layer
        :param layer_id: (int)
        :return: layer (Sequential)
        """

        layer = Sequential()
        layer.add_module("fc",
                         Linear(self.dims[layer_id],
                                self.dims[layer_id + 1],
                                bias=self.bias[layer_id]))
        if self.use_bn[0]:
            bn = nn.BatchNorm1d(self.dims[layer_id + 1])
            layer.add_module("bn", bn)

        layer.add_module("dropout", nn.Dropout(self.dropout[layer_id]))

        if act:
            layer.add_module(self.act, self.activation)
        return layer

    def _set_biases(self, bias, num_layers):
        """
        Set and control bias input
        """
        if isinstance(bias, bool):
            self.bias = num_layers * [bias]
        elif isinstance(bias, list):
            assert len(
                bias
            ) == num_layers, "Length of bias should match the number of layers."
            self.bias = bias
        else:
            raise ValueError(
                "Unsupported type for bias. Needs to be of type bool or list.")

    def _set_batch_norm(self, use_bn, num_layers):
        """
        Set and control batch norm param
        """
        if isinstance(use_bn, bool):
            self.use_bn = num_layers * [use_bn]
        else:
            raise ValueError(
                "Unsupported type for batch norm. Needs to be of type bool.")

    def _set_dropout(self, dropout, num_layers):
        """
        Set and control dropout params
        """
        if isinstance(dropout, float):
            self.dropout = num_layers * [dropout]
        else:
            raise ValueError(
                "Unsupported type for batch norm. Needs to be of type float.")

    def _set_mlp_dimensions(self, in_dim, h_dim, out_dim, num_layers):
        """
        Set and control mlp dimensions
        """
        if isinstance(h_dim, int):
            self.dims = [in_dim] + (num_layers - 1) * [h_dim] + [out_dim]
        elif isinstance(h_dim, list):
            assert len(h_dim) == (
                num_layers - 1
            ), "Length of h_dim should match the number of hidden layers."
            self.dims = [in_dim] + h_dim + [out_dim]
        else:
            raise ValueError(
                "Unsupported type for h_dim. Needs to be int or list."
            )

    def _set_activations(self, act):
        """
        Set and control activations
        """
        if act in ACTIVATIONS.keys():
            self.act = act
            self.activation = ACTIVATIONS[act]
        else:
            raise ValueError(
                'Unsupported type of activation function: {}. Choose among {}'.
                format(act, list(ACTIVATIONS.keys()))
            )

    def set_lrp(self, with_lrp):
        self.with_lrp = with_lrp
        if self.with_lrp:
            self.forward_activations = []

    def forward(self, feats):
        """
        MLP forward
        :param feats: (FloatTensor) features to pass through MLP
        :return: out: MLP output
        """
        out = feats
        if hasattr(self, 'with_lrp') and self.with_lrp:
            self.forward_activations.append(out)
        for layer in self.mlp:
            out = layer(out)
            if hasattr(self, 'with_lrp') and self.with_lrp:
                self.forward_activations.append(out)
        return out

    def lrp(self, relevance_score):
        for layer_id in range(len(self.mlp) - 1, -1, -1):
            pos_weights = torch.clamp(self.mlp[layer_id][0].weight, min=0)
            rel_unnorm = torch.mm(
                self.forward_activations[layer_id],
                pos_weights.t()) + 1e-9
            rel_unnorm = relevance_score / rel_unnorm
            contrib = torch.mm(rel_unnorm, pos_weights)
            relevance_score = self.forward_activations[layer_id] * contrib
        return relevance_score
