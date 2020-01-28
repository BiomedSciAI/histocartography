from histocartography.ml.layers.mlp import MLP
from histocartography.ml.models.base_model import BaseModel
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN


class SuperpxGraphModel(BaseModel):
    """
    Superpx Graph Model. Apply a GNN at the super pixel graph level.
    """

    def __init__(self, config, node_dim):
        """
        SuperpxGraphMddel model constructor
        :param config: (dict) configuration parameters
        :param node_dim: (int) superpx dim, data specific argument
        """

        super(SuperpxGraphModel, self).__init__(config)

        # 1- set class attributes
        self.config = config
        self.hl_node_dim = node_dim
        self.gnn_params = config['gnn_params']['superpx_gnn']
        self.readout_params = self.config['readout']

        # 2- build cell graph params
        self._build_superpx_graph_params(
            self.gnn_params,
            input_dim=self.hl_node_dim
        )

        # 3- build classification params
        self._build_classification_params()

    def _build_classification_params(self):
        """
        Build classification parameters
        """
        if self.concat:
            emd_dim = self.gnn_params['input_dim'] + \
                self.gnn_params['hidden_dim'] * (self.gnn_params['n_layers'] - 1) + \
                self.gnn_params['output_dim']
        else:
            emd_dim = self.gnn_params['output_dim']

        self.pred_layer = MLP(in_dim=emd_dim,
                              h_dim=self.readout_params['hidden_dim'],
                              out_dim=self.num_classes,
                              num_layers=self.readout_params['num_layers']
                              )

    def forward(self, data):
        """
        Foward pass.
        :param superpx_graph: (DGLGraph) superpx graph
        """

        # 1. GNN layers over the high level graph (super pixel graph)
        superpx_graph = data[0]
        feats = superpx_graph.ndata[GNN_NODE_FEAT_IN]
        graph_embeddings = self.superpx_gnn(superpx_graph, feats, self.concat)

        # 2. Run readout function
        logits = self.pred_layer(graph_embeddings)
        return logits
