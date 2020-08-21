#from histocartography.interpretability.base_explainer import BaseExplainer

from ..base_explainer import BaseExplainer

import numpy as np
import torch
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
        self.gnn_layer = '0'       # gnn_layer (str): name of the last GNN layer -- extract from the config file 

        # # debug purposes 
        # m2 = self.model.cell_graph_gnn.layers._modules.get('2')
        # print('m2', m2)

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

        self.extractor = GradCAM(self.model.cell_graph_gnn.layers, self.gnn_layer)

        # 2/ forward pass
        logits = self.model(data)

        print('Logits', logits)
        print('Label', label)

        # 3/ extract node importance 
        node_importance = self.extractor(label, logits).cpu()
        graph.ndata['node_importance'] = torch.sum(node_importance, dim=1)
        self.extractor.clear_hooks()

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
























