import numpy as np
import torch
from copy import deepcopy
from torch import nn
import dgl 

from .grad_cam import GradCAMpp
from ..base_explainer import BaseExplainer


class GraphGradCAMPPExplainer(BaseExplainer):
    def __init__(self, **kwargs) -> None:
        """
        Constructor for GradCAM++ explainer.
        """
        super(GraphGradCAMPPExplainer, self).__init__(**kwargs)

        all_param_names = [name for name, _ in self.model.named_parameters()]
        self.gnn_layer_ids = list(set([p.split('.')[2] for p in all_param_names]))
        self.gnn_layer_name = all_param_names[0].split('.')[0]

    def process(self, graph):
        """
        Explain a graph. 

        Args:
            graph (dgl.DGLGraph): graph to explain. 
        """

        all_node_importances = []
        for layer_id in self.gnn_layer_ids:
            self.extractor = GradCAMpp(getattr(self.model, self.gnn_layer_name).layers, layer_id)
            original_logits = self.model([deepcopy(graph)])
            winning_class = original_logits.argmax().item()
            node_importance = self.extractor(winning_class, original_logits).cpu()
            all_node_importances.append(node_importance)
            self.extractor.clear_hooks()
            
        node_importance = torch.stack(all_node_importances, dim=1).mean(dim=1)
        node_importance = node_importance.cpu().detach().numpy()

        logits = original_logits.cpu().detach().numpy()

        return node_importance, logits
