#from histocartography.interpretability.base_explainer import BaseExplainer

import numpy as np
import torch
from copy import deepcopy
from torch import nn
import dgl 
import networkx as nx 
import gc

from histocartography.interpretability.saliency_explainer.grad_cam import GradCAMpp, GradCAM
from histocartography.interpretability.constants import KEEP_PERCENTAGE_OF_NODE_IMPORTANCE, MODEL_TYPE_TO_GNN_LAYER_NAME
from ..explanation import GraphExplanation
from ..base_explainer import BaseExplainer
from histocartography.utils.torch import torch_to_list, torch_to_numpy


class GraphGradCAMPPExplainer(BaseExplainer):
    def __init__(
            self,
            model,
            config,
            cuda=False,
            verbose=False
    ):
        """
        GradCAM++ for GNN-based saliency explanation constructor
        :param model: (nn.Module) a pre-trained model to run the forward pass
        :param config: (dict) method-specific parameters
        :param cuda: (bool) if cuda is enable
        :param verbose: (bool) if verbose is enable
        """
        super(GraphGradCAMPPExplainer, self).__init__(model, config, cuda, verbose)

        self.gnn_layer_ids = ['0', '1', '2']  # @TODO read from the config file
        self.gnn_layer_name = MODEL_TYPE_TO_GNN_LAYER_NAME[config['model_params']['model_type']]

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
        self.model.set_forward_hook(self.model.pred_layer.mlp, '0')  # hook before the last classification layer for extracting latent representation

        # 2/ forward-pass -- applying avgGradCAM++ (avg over all the GNN layers)
        all_node_importances = []
        for layer_id in self.gnn_layer_ids:
            self.extractor = GradCAMpp(getattr(self.model, self.gnn_layer_name).layers, layer_id)
            original_logits = self.model([deepcopy(graph)])
            winning_class = original_logits.argmax().item()
            node_importance = self.extractor(winning_class, original_logits).cpu()
            all_node_importances.append(node_importance)
            self.extractor.clear_hooks()
            
        graph.ndata['node_importance'] = torch.stack(all_node_importances, dim=1).mean(dim=1)

        # 3/ prune the graph using the node importance -- then forward again and store the logits/latent representation
        explanation_graphs = {}
        for keep_percentage in KEEP_PERCENTAGE_OF_NODE_IMPORTANCE:
            # a. prune graph
            pruned_graph = self._build_pruned_graph(graph, keep_percentage)
            # b. forward pass
            logits = self.model([pruned_graph])
            # c. store in dict 
            explanation_graphs[keep_percentage] = {}
            explanation_graphs[keep_percentage]['logits'] = torch_to_list(logits.squeeze())
            explanation_graphs[keep_percentage]['latent'] = torch_to_list(self.model.latent_representation.squeeze())
            explanation_graphs[keep_percentage]['num_nodes'] = pruned_graph.number_of_nodes()
            explanation_graphs[keep_percentage]['num_edges'] = pruned_graph.number_of_edges()
            explanation_graphs[keep_percentage]['node_importance'] = torch_to_list(pruned_graph.ndata['node_importance'])
            explanation_graphs[keep_percentage]['centroid'] = torch_to_list(pruned_graph.ndata['centroid'])
            explanation_graphs[keep_percentage]['nuclei_label'] = torch_to_list(pruned_graph.ndata['nuclei_label'])
            explanation_graphs[keep_percentage]['node_idx_to_keep'] = torch_to_list(pruned_graph.ndata['node_idx_to_keep'])
            if self.store_instance_map:
                explanation_graphs[keep_percentage]['instance_map'] = torch_to_list(data[3][0])

        # 4/ build and return explanation 
        explanation = GraphExplanation(
            self.config,
            image[0],
            image_name[0],
            label,
            explanation_graphs,
        )

        return explanation
