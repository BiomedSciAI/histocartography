#from histocartography.interpretability.base_explainer import BaseExplainer

from ..base_explainer import BaseExplainer

import numpy as np
import torch
from copy import deepcopy
from torch import nn
from histocartography.interpretability.saliency_explainer.grad_cam import GradCAM
from ..explanation import GraphExplanation


class GraphGradCAMExplainer(BaseExplainer):
    def __init__(
            self,
            model,
            config,
            cuda=False,
            verbose=False
    ):
        """
        CAM for CNN-based saliency explanation constructor
        :param model: (nn.Module) a pre-trained model to run the forward pass
        :param config: (dict) method-specific parameters
        :param cuda: (bool) if cuda is enable
        :param verbose: (bool) if verbose is enable
        """
        super(GraphGradCAMExplainer, self).__init__(model, config, cuda, verbose)

        # Set based on our trained CNN-single stream (10x)-ResNet34 network
        self.gnn_layer_ids = ['0', '1', '2']

    def explain(self, data, label):
        """
        Explain a image patch instance
        :param data: list with a graph, an image, an image name 
        :param label: (int) label for the input data
        """

        # 1/ pre-processing
        graph = data[0]
        image = data[1]
        image_name = data[2]
        if self.cuda:
            self.model = self.model.cuda()
        self.model.eval()
        self.model.zero_grad()

        all_node_importances = []
        for layer_id in self.gnn_layer_ids:
            self.extractor = GradCAM(self.model.cell_graph_gnn.layers, layer_id)
            logits = self.model([deepcopy(graph)])
            node_importance = self.extractor(label, logits).cpu()
            all_node_importances.append(torch.sum(node_importance, dim=1))
            self.extractor.clear_hooks()

        graph.ndata['node_importance'] = torch.stack(all_node_importances, dim=1).mean(dim=1)

        # 4/ build and return explanation 
        explanation = GraphExplanation(
            self.config,
            image,
            image_name,
            logits,
            label,
            graph,
        )

        return explanation
























