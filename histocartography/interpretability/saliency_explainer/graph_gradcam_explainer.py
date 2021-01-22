import numpy as np
import torch
from copy import deepcopy
from torch import nn
import dgl 

from .grad_cam import GradCAM
from ..base_explainer import BaseExplainer


class GraphGradCAMExplainer(BaseExplainer):
    def __init__(self, **kwargs) -> None:
        """
        GradCAM explainer constructor. 
        """
        super(GraphGradCAMExplainer, self).__init__(**kwargs)

        all_param_names = [name for name, _ in self.model.named_parameters()]
        self.gnn_layer_ids = list(set([p.split('.')[2] for p in all_param_names]))
        self.gnn_layer_name = all_param_names[0].split('.')[0]

    def process(self, graph, class_idx=None):
        """
        Explain a graph. 

        Args:
            graph (dgl.DGLGraph): graph to explain. 
            class_idx (int, Optional): Index of the class to explain. If None, explainer winning class. 
        
        Returns:
            node_importance (np.ndarray): Node-level importance scores
            logits (np.ndarray): Prediction logits 
        """

        all_node_importances = []
        for layer_id in self.gnn_layer_ids:
            self.extractor = GradCAM(getattr(self.model, self.gnn_layer_name).layers, layer_id)
            original_logits = self.model([deepcopy(graph)])
            if class_idx is None:
                class_idx = original_logits.argmax().item()
            node_importance = self.extractor(class_idx, original_logits).cpu()
            all_node_importances.append(node_importance)
            self.extractor.clear_hooks()

        node_importance = torch.stack(all_node_importances, dim=1).mean(dim=1)
        node_importance = node_importance.cpu().detach().numpy()

        logits = original_logits.cpu().detach().numpy()

        return node_importance, logits
