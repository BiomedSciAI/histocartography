import dgl

from histocartography.ml.layers.mlp import MLP
from histocartography.ml.models.base_model import BaseModel
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN


class CellGraphModel(BaseModel):
    """
    Cell Graph Model. Apply a GNN at the cell graph level.
    """

    def __init__(self, config, node_dim):
        """
        CellGraphModel model constructor
        :param config: (dict) configuration parameters
        :param node_dim: (int) cell dim, data specific argument
        """

        super(CellGraphModel, self).__init__(config)

        # 1- set class attributes
        self.config = config
        self.ll_node_dim = node_dim
        self.gnn_params = config['gnn_params']['cell_gnn']
        self.readout_params = self.config['readout']

        # 2- build cell graph params
        self._build_cell_graph_params(self.gnn_params)

        # 3- build classification params
        self._build_classification_params()

    def _build_classification_params(self):
        """
        Build classification parameters
        """
        if self.readout_agg_op == "concat":
            # emd_dim = self.gnn_params['input_dim'] + \
            #     self.gnn_params['hidden_dim'] * (self.gnn_params['n_layers'] - 1) + \
            #     self.gnn_params['output_dim']
            emd_dim = self.gnn_params['hidden_dim'] * (self.gnn_params['n_layers'] - 1) + \
                self.gnn_params['output_dim']
        else:
            emd_dim = self.gnn_params['output_dim']

        self.pred_layer = MLP(
            in_dim=emd_dim,
            h_dim=self.readout_params['hidden_dim'],
            out_dim=self.num_classes,
            num_layers=self.readout_params['num_layers']
        )

    def forward(self, data):
        """
        Foward pass.
        :param data: tuple with (DGLGraph), cell graph
        """

        if isinstance(data[0], dgl.DGLGraph):
            # 1. GNN layers over the low level graph
            cell_graph = data[0]
            feats = cell_graph.ndata[GNN_NODE_FEAT_IN]
            graph_embeddings = self.cell_graph_gnn(cell_graph, feats)
        else:
            adj, feats = data[0], data[1]
            graph_embeddings = self.cell_graph_gnn(adj, feats)

        # 2. Run readout function
        logits = self.pred_layer(graph_embeddings)

        return logits
