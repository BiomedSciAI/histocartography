import dgl
from typing import Dict

from ..layers.mlp import MLP
from .base_model import BaseModel
from .. import MultiLayerGNN
from ..layers.constants import GNN_NODE_FEAT_IN


class TissueGraphModel(BaseModel):
    """
    Tissue Graph Model. Apply a GNN on tissue level.
    """

    def __init__(
        self,
        gnn_params,
        classification_params,
        node_dim,
        **kwargs):
        """
        TissueGraphModel model constructor

        Args:
            gnn_params: (dict) GNN configuration parameters.
            classification_params: (dict) classification configuration parameters.
            node_dim (int): Tissue node feature dimension. 
        """

        super().__init__(**kwargs)

        # 1- set class attributes
        self.node_dim = node_dim  
        self.gnn_params = gnn_params
        self.classification_params = classification_params
        self.readout_op = gnn_params['readout_op']

        # 2- build tissue graph params
        self._build_tissue_graph_params()

        # 3- build classification params
        self._build_classification_params()

    def _build_tissue_graph_params(self):
        """
        Build multi layer GNN for tissue processing.
        """
        self.superpx_gnn = MultiLayerGNN(
            input_dim=self.node_dim,
            **self.gnn_params
        )

    def _build_classification_params(self):
        """
        Build classification parameters
        """
        if self.readout_op == "concat":
            emd_dim = self.gnn_params['hidden_dim'] * (
                self.gnn_params['n_layers'] - 1) + self.gnn_params['output_dim']
        else:
            emd_dim = self.gnn_params['output_dim']

        self.pred_layer = MLP(in_dim=emd_dim,
                              h_dim=self.classification_params['hidden_dim'],
                              out_dim=self.num_classes,
                              num_layers=self.classification_params['num_layers']
                              )

    def forward(self, data):
        """
        Foward pass.
        :param superpx_graph: (DGLGraph) superpx graph
        @TODO: input can be:
            - DGLGraph
            - [DGLGraph]
            - [tensor (adj), tensor (node features)]
        """

        if isinstance(data, dgl.DGLGraph) or isinstance(data[0], dgl.DGLGraph):
            # 1. GNN layers over the low level graph
            if isinstance(data, list):
                superpx_graph = data[0]
            else:
                superpx_graph = data
            feats = superpx_graph.ndata[GNN_NODE_FEAT_IN]
            graph_embeddings = self.superpx_gnn(superpx_graph, feats)
        else:
            adj, feats = data[0], data[1]
            graph_embeddings = self.superpx_gnn(adj, feats)

        # 2. Run readout function
        logits = self.pred_layer(graph_embeddings)
        return logits

    def set_lrp(self, with_lrp):
        self.superpx_gnn.set_lrp(with_lrp)
        self.pred_layer.set_lrp(with_lrp)

    def lrp(self, out_relevance_score):
        # lrp over the classification
        relevance_score = self.pred_layer.lrp(out_relevance_score)

        # lrp over the GNN layers
        relevance_score = self.superpx_gnn.lrp(relevance_score)

        return relevance_score
