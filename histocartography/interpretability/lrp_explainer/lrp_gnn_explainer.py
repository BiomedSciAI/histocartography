import torch
from copy import deepcopy

from histocartography.utils.io import get_device
from histocartography.interpretability.constants import KEEP_PERCENTAGE_OF_NODE_IMPORTANCE
from ..base_explainer import BaseExplainer
from histocartography.utils.torch import torch_to_list, torch_to_numpy


class LRPGNNExplainer(BaseExplainer):

    def _apply_rlp(self, graph):
        logits = self.model([deepcopy(graph)]).squeeze()
        max_idx = logits.argmax(dim=0)
        init_relevance = torch.zeros_like(logits)
        init_relevance[max_idx] = logits[max_idx]
        node_importance = self.model.rlp(init_relevance)
        node_importance = torch.sum(node_importance, dim=1)
        return node_importance, logits

    def process(self, graph):
        """
        Explain a graph. 

        Args:
            graph (dgl.DGLGraph): graph to explain. 
        """

        self.model.zero_grad()
        self.model.set_forward_hook(self.model.pred_layer.mlp, '0')  # hook before the last classification layer
        self.model.set_rlp(True)

        node_importance, logits = self._apply_rlp(graph)

        node_importance = node_importance.cpu().detach().numpy()
        logits = logits.cpu().detach().numpy()

        return node_importance, logits