import torch
import torch.nn as nn
import importlib
import dgl

from histocartography.ml.layers.constants import (
    AVAILABLE_LAYER_TYPES, GNN_MODULE,
    GNN_NODE_FEAT_OUT, READOUT_TYPES,
    REDUCE_TYPES
)


class MultiLayerGNN(nn.Module):
    """
    MultiLayer network that concatenates several gnn layers.
    """

    def __init__(
        self,
        layer_type="gin_layer",
        input_dim=None,
        output_dim=32,
        num_layers=3,
        readout_op="concat",
        readout_type="mean",
        **kwargs
    ) -> None:
        """
        MultiLayer GNN constructor.

        Args:
            layer_type (str): GNN layer type. Default to "gin_layer".
            input_dim (int): Input dimension of the node features. Default to None.
            output_dim (int): Output dimension of the node embeddings. Default to 32.
            num_layers (int): Number of GNN layers. Default to 3.
            readout_op (str): How the intermediate node embeddings are aggregated. Default to "concat".
            readout_type (str): Global node pooling operation. Default to "mean".
        """

        assert input_dim is not None, "Please provide input node dimensions."

        super(MultiLayerGNN, self).__init__()

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

        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.output_dim = output_dim
        self.readout_op = readout_op
        self.readout_type = readout_type

        # input layer
        self.layers.append(getattr(module, AVAILABLE_LAYER_TYPES[layer_type])(
            node_dim=input_dim,
            out_dim=output_dim,
            **kwargs
        )
        )
        # hidden layers
        for i in range(1, num_layers - 1):
            self.layers.append(
                getattr(
                    module,
                    AVAILABLE_LAYER_TYPES[layer_type])(
                    node_dim=output_dim,
                    out_dim=output_dim,
                    **kwargs
                )
            )
        # output layer
        self.layers.append(getattr(module, AVAILABLE_LAYER_TYPES[layer_type])(
            node_dim=output_dim,
            out_dim=output_dim,
            **kwargs
        )
        )

        # readout op
        if readout_op == "lstm":
            self.lstm = nn.LSTM(
                output_dim, (num_layers * output_dim) // 2,
                bidirectional=True,
                batch_first=True)
            self.att = nn.Linear(2 * ((num_layers * output_dim) // 2), 1)

        # set kwargs as arguments for model identification
        for arg, val in kwargs.items():
            setattr(self, arg, val)

    def forward(self, g, h, with_readout=True):
        """
        Forward pass.
        :param g: (DGLGraph)
        :param h: (FloatTensor)
        :param cat: (bool) if concat the features at each conv layer
        :return:
        """
        h_concat = []
        for layer in self.layers:
            h = layer(g, h)
            h_concat.append(h)

        if isinstance(g, dgl.DGLGraph):

            # aggregate the multi-scale node representations
            if self.readout_op == "concat":
                g.ndata[GNN_NODE_FEAT_OUT] = torch.cat(h_concat, dim=-1)
            elif self.readout_op == "lstm":
                # [num_nodes, num_layers, num_channels]
                x = torch.stack(h_concat, dim=1)
                alpha, _ = self.lstm(x)
                alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
                alpha = torch.softmax(alpha, dim=-1)
                g.ndata[GNN_NODE_FEAT_OUT] = (
                    x * alpha.unsqueeze(-1)).sum(dim=1)
            elif self.readout_op == "none":
                g.ndata[GNN_NODE_FEAT_OUT] = h
            else:
                raise ValueError(
                    "Unsupported readout operator. Options are 'concat', 'lstm', 'none'.")

            # readout
            if with_readout:
                return READOUT_TYPES[self.readout_type](g, GNN_NODE_FEAT_OUT)

            return g.ndata.pop(GNN_NODE_FEAT_OUT)

        else:
            if self.readout_op == "concat":
                h_concat = [h.squeeze() for h in h_concat]
                h = torch.cat(h_concat, dim=-1)
            elif self.readout_op == "lstm":
                raise NotImplementedError(
                    "LSTM aggregation with LSTM is not supported.")

            # readout
            if with_readout:
                return REDUCE_TYPES[self.readout_type](h, dim=0)
            return h

    def set_lrp(self, with_lrp):
        for layer in self.layers:
            layer.set_lrp(with_lrp)

    def lrp(self, relevance_score):
        for layer_id in range(len(self.layers) - 1, -1, -1):
            relevance_score = self.layers[layer_id].lrp(relevance_score)
        return relevance_score
