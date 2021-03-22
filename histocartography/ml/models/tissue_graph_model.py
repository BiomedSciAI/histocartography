import dgl 
from typing import Dict

from histocartography.ml.layers.mlp import MLP
from histocartography.ml.models.base_model import BaseModel
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN


class TissueGraphModel(BaseModel):
    """
    Tissue Graph Model. Apply a GNN on tissue level.
    """

    # def __init__(self, config: Dict, input_node_dim: int):
    def __init__(self, gnn_params, input_node_dim: int):
        """
        TissueGraphModel model constructor.
        :param config: (dict) configuration parameters.
        :param input_node_dim (int): feature dim of each node. 
        """

        super(TissueGraphModel, self).__init__(config)

        # 1- set class attributes
        self.config = config
        self.hl_node_dim = input_node_dim  # @TODO: rename hl node dim to node dim 
        self.gnn_params = config['gnn_params']['superpx_gnn']
        self.readout_params = self.config['readout']
        self.readout_agg_op = config['gnn_params']['superpx_gnn']['agg_operator']

        # 2- build tissue graph params
        self._build_tissue_graph_params(
            self.gnn_params,
            input_dim=self.hl_node_dim
        )

        # 3- build classification params
        self._build_classification_params()

    def _build_classification_params(self):
        """
        Build classification parameters
        """
        if self.readout_agg_op == "concat":
            emd_dim = self.gnn_params['hidden_dim'] * (self.gnn_params['n_layers'] - 1) + \
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

