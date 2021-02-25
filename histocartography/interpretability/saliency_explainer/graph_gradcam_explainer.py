from copy import deepcopy
from typing import List, Optional, Tuple

import dgl
import numpy as np
import torch

from ..base_explainer import BaseExplainer
from .grad_cam import GradCAM


class GraphGradCAMExplainer(BaseExplainer):
    def __init__(self, **kwargs) -> None:
        """
        GradCAM explainer constructor.
        """
        super().__init__(**kwargs)
        all_param_names = [name for name, _ in self.model.named_parameters()]
        self.gnn_layer_ids = list(
            filter(
                lambda x: x.isdigit(), set([p.split(".")[2] for p in all_param_names])
            )
        )
        self.gnn_layer_name = all_param_names[0].split(".")[0]

    def process(
        self, graph: dgl.DGLGraph, class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute node importances for a single class

        Args:
            graph (dgl.DGLGraph): Graph to explain
            class_idx (Optional[int], optional): Class index to explain. None results in using the winning class. Defaults to None.

        Returns:
            node_importance (np.ndarray): Node-level importance scores
            logits (np.ndarray): Prediction logits
        """
        node_importances, logits = self.process_all(graph, [class_idx])
        return node_importances.unsqueeze(0), logits

    def process_all(
        self, graph: dgl.DGLGraph, classes: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute node importances for all classes

        Args:
            graph (dgl.DGLGraph): Graph to explain
            classes (List[int]): Classes to explain

        Returns:
            node_importance (np.ndarray): Node-level importance scores
            logits (np.ndarray): Prediction logits
        """
        graph_copy = dgl.DGLGraph(graph_data=graph)
        for k, v in graph.ndata.items():
            graph_copy.ndata[k] = v.clone()
        for k, v in graph.edata.items():
            graph_copy.edata[k] = v.clone()
        #graph_copy = deepcopy(graph)
        self.extractor = GradCAM(
            getattr(self.model, self.gnn_layer_name).layers, self.gnn_layer_ids
        )
        original_logits = self.model(graph_copy)
        if isinstance(original_logits, tuple):
            original_logits = original_logits[0]
        if classes[0] is None:
            classes = [original_logits.argmax().item()]
        all_class_importances = list()
        for class_idx in classes:
            node_importance = self.extractor(
                class_idx, original_logits, normalized=True
            ).cpu()
            all_class_importances.append(node_importance)
            self.extractor.clear_hooks()
        logits = original_logits.cpu().detach().numpy()
        node_importances = torch.stack(all_class_importances)
        node_importances = node_importances.cpu().detach().numpy()
        return node_importances, logits
