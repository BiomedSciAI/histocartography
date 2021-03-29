import dgl

from ..layers.mlp import MLP
from .base_model import BaseModel
from .. import MultiLayerGNN
from ..layers.constants import GNN_NODE_FEAT_IN


class CellGraphModel(BaseModel):
    """
    Cell Graph Model. Apply a GNN at the cell graph level.
    """

    def __init__(
        self,
        gnn_params,
        classification_params,
        node_dim,
        **kwargs):
        """
        CellGraphModel model constructor

        Args:
            gnn_params: (dict) GNN configuration parameters.
            classification_params: (dict) classification configuration parameters.
            node_dim (int): Cell node feature dimension. 
        """

        super().__init__(**kwargs)

        # 1- set class attributes
        self.node_dim = node_dim
        self.gnn_params = gnn_params
        self.readout_op = gnn_params['readout_op']
        self.classification_params = classification_params

        # 2- build cell graph params
        self._build_cell_graph_params()

        # 3- build classification params
        self._build_classification_params()

    def _build_cell_graph_params(self):
        """
        Build cell graph multi layer GNN
        """
        self.cell_graph_gnn = MultiLayerGNN(
            input_dim=self.node_dim,
            **self.gnn_params
            )

    def _build_classification_params(self):
        """
        Build classification parameters
        """
        if self.readout_op == "concat":
            emd_dim = self.gnn_params['output_dim'] * self.gnn_params['num_layers']
        else:
            emd_dim = self.gnn_params['output_dim']

        self.pred_layer = MLP(
            in_dim=emd_dim,
            h_dim=self.classification_params['hidden_dim'],
            out_dim=self.num_classes,
            num_layers=self.classification_params['num_layers']
        )

    def forward(self, data):
        """
        Foward pass.
        :param data: tuple with (DGLGraph), cell graph
        """

        if isinstance(data, dgl.DGLGraph) or isinstance(data[0], dgl.DGLGraph):
            # 1. GNN layers over the low level graph
            if isinstance(data, list):
                cell_graph = data[0]
            else:
                cell_graph = data
            feats = cell_graph.ndata[GNN_NODE_FEAT_IN]
            graph_embeddings = self.cell_graph_gnn(cell_graph, feats)
        else:
            adj, feats = data[0], data[1]
            graph_embeddings = self.cell_graph_gnn(adj, feats)

        # 2. Run readout function
        out = self.pred_layer(graph_embeddings)

        return out

    def set_lrp(self, with_lrp):
        self.cell_graph_gnn.set_lrp(with_lrp)
        self.pred_layer.set_lrp(with_lrp)

    def lrp(self, out_relevance_score):
        # lrp over the classification
        relevance_score = self.pred_layer.lrp(out_relevance_score)

        # lrp over the GNN layers
        relevance_score = self.cell_graph_gnn.lrp(relevance_score)

        return relevance_score
