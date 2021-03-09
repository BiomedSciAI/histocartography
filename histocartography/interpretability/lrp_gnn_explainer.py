import torch
from copy import deepcopy
import dgl 

from .base_explainer import BaseExplainer
from ..utils.torch import torch_to_numpy


class LRPGNNExplainer(BaseExplainer):
    """
    Layerwise-Relevance Propagation. This module will only work
    if the model was built with the ml library provided. 
    """

    def _apply_rlp(self, graph):
        logits = self.model([deepcopy(graph)]).squeeze()
        max_idx = logits.argmax(dim=0)
        init_relevance = torch.zeros_like(logits)
        init_relevance[max_idx] = logits[max_idx]
        node_importance = self.model.rlp(init_relevance)
        node_importance = torch.sum(node_importance, dim=1)
        return node_importance, logits

    def process(self, graph: dgl.DGLGraph):
        """
        Explain a graph with LRP. 

        Args:
            graph (dgl.DGLGraph): graph to explain. 

        Returns:
            node_importance (np.ndarray): Node importance scores. 
            logits (np.ndarray): Predicted logits. 
        """

        self.model.zero_grad()
        self.model.set_forward_hook(self.model.pred_layer.mlp, '0')  # hook before the last classification layer
        self.model.set_rlp(True)

        node_importance, logits = self._apply_rlp(graph)

        node_importance = torch_to_numpy(node_importance)
        logits = torch_to_numpy(logits)

        return node_importance, logits