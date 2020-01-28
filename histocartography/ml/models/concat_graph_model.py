import torch

from histocartography.ml.models.base_model import BaseModel
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN
from histocartography.ml.layers.mlp import MLP


class ConcatGraphModel(BaseModel):
    """
    Concat graph model. The information for grading tumors in WSI lies at different scales. By building 2 graphs,
    one at the cell level and one at the object level (modeled with super pixels), we can extract graph embeddings
    that once combined provide a multi-scale representation of a tumor. The combination in this model is done simply
    by concatenating the graph embeddings.

    This implementation is using GIN Layers as a graph neural network.

    """

    def __init__(self, config, node_dims):
        """
        MultiLevelGraph model constructor
        :param config: (dict) configuration parameters
        :param node_dims: (typle of int) low level and high level node dim, data specific argument
        """

        super(ConcatGraphModel, self).__init__(config)

        # 1- set class attributes
        self.config = config
        self.ll_node_dim = node_dims[0]
        self.hl_node_dim = node_dims[1]

        # 2- build cell graph params
        self._build_cell_graph_params(config['gnn_params'][0])

        # 3- build super pixel graph params
        self._build_superpx_graph_params(config['gnn_params'][1], input_dim=self.hl_node_dim)

        # 4- build classification params
        self._build_classification_params()

    def _build_classification_params(self):
        """
        Build classification parameters
        """
        if self.concat:
            emd_dim = self.config['gnn_params'][0]['input_dim'] + \
                self.config['gnn_params'][0]['hidden_dim'] * (self.config['gnn_params'][0]['n_layers'] - 1) + \
                self.config['gnn_params'][0]['output_dim'] + \
                self.config['gnn_params'][1]['input_dim'] + \
                self.config['gnn_params'][1]['hidden_dim'] * (self.config['gnn_params'][1]['n_layers'] - 1) + \
                self.config['gnn_params'][1]['output_dim']
        else:
            emd_dim = self.config['gnn_params'][0]['output_dim'] + \
                      self.config['gnn_params'][1]['output_dim']

        self.pred_layer = MLP(
            in_dim=emd_dim,
            h_dim=self.config['readout']['hidden_dim'],
            out_dim=self.num_classes,
            num_layers=self.config['readout']['num_layers']
        )

    def _update_config(self, config, input_dim=None):
        """
        Update config params with data-dependent parameters
        """
        if input_dim is not None:
            config['input_dim'] = input_dim

        config['use_bn'] = self.use_bn

    def forward(self, data):
        """
        Foward pass.
        :param data: tuple of: - (DGLGraph) low level graph,
                               - (DGLGraph) high level graph,
        """

        cell_graph, superpx_graph = data[0], data[1]

        # 1. GNN layers over the low level graph
        ll_feats = cell_graph.ndata[GNN_NODE_FEAT_IN]
        ll_graph_emb = self.cell_graph_gnn(cell_graph, ll_feats, self.concat)

        # 3. GNN layers over the high level graph
        hl_feats = superpx_graph.ndata[GNN_NODE_FEAT_IN]
        hl_graph_emb = self.superpx_gnn(superpx_graph, hl_feats, self.concat)

        # 4. Classification layers
        graph_embeddings = torch.cat((ll_graph_emb, hl_graph_emb), dim=1)

        logits = self.pred_layer(graph_embeddings)

        return logits
