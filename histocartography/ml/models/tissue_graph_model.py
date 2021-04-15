import dgl
from typing import Dict, Union, Tuple
import torch
import os

from ..layers.mlp import MLP
from .base_model import BaseModel
from .. import MultiLayerGNN
from ..layers.constants import GNN_NODE_FEAT_IN
from .zoo import MODEL_NAME_TO_URL, MODEL_NAME_TO_CONFIG
from ...utils import download_box_link


class TissueGraphModel(BaseModel):
    """
    Tissue Graph Model. Apply a GNN on tissue level.
    """

    def __init__(
            self,
            gnn_params: Dict,
            classification_params: Dict,
            node_dim: int,
            **kwargs):
        """
        TissueGraphModel model constructor.

        Args:
            gnn_params (Dict): GNN configuration parameters.
            classification_params (Dict): classification configuration parameters.
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

        # 4- load pretrained weights if needed
        if self.pretrained:
            model_name = self._get_checkpoint_id()
            if model_name:
                self._load_checkpoint(model_name)
            else:
                raise NotImplementedError(
                    'There is not available TG-GNN checkpoint for the provided params.')

    def _get_checkpoint_id(self):

        # 1st level-check: Model type, GNN layer type, num classes
        model_type = 'tggnn'
        layer_type = self.gnn_params['layer_type'].replace('_layer', '')
        num_classes = self.num_classes
        candidate = 'bracs_' + model_type + '_' + \
            str(num_classes) + '_classes_' + layer_type + '.pt'
        if candidate not in list(MODEL_NAME_TO_URL.keys()):
            return ''

        # 2nd level-check: Look at all the specific params
        cand_config = MODEL_NAME_TO_CONFIG[candidate]

        for cand_key, cand_val in cand_config['gnn_params'].items():
            if hasattr(self.superpx_gnn, cand_key):
                if cand_val != getattr(self.superpx_gnn, cand_key):
                    return ''
            else:
                if cand_val != getattr(self.superpx_gnn.layers[0], cand_key):
                    return ''

        for cand_key, cand_val in cand_config['classification_params'].items():
            if cand_val != getattr(self.pred_layer, cand_key):
                return ''

        if cand_config['node_dim'] != self.node_dim:
            return ''

        return candidate

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
            emd_dim = self.gnn_params['output_dim'] * \
                self.gnn_params['num_layers']
        else:
            emd_dim = self.gnn_params['output_dim']

        self.pred_layer = MLP(
            in_dim=emd_dim,
            hidden_dim=self.classification_params['hidden_dim'],
            out_dim=self.num_classes,
            num_layers=self.classification_params['num_layers'])

    def forward(
        self,
        graph: Union[dgl.DGLGraph, Tuple[torch.tensor, torch.tensor]]
    ) -> torch.tensor:
        """
        Foward pass.

        Args:
            graph (Union[dgl.DGLGraph, Tuple[torch.tensor, torch.tensor]]): Tissue graph to process.

        Returns:
            torch.tensor: Model output.
        """

        # 1. GNN layers over the tissue graph
        if isinstance(graph, dgl.DGLGraph):
            feats = graph.ndata[GNN_NODE_FEAT_IN]
            graph_embeddings = self.superpx_gnn(graph, feats)
        else:
            adj, feats = graph[0], graph[1]
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
