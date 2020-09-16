import torch
from copy import deepcopy

from histocartography.utils.io import get_device
from histocartography.interpretability.constants import KEEP_PERCENTAGE_OF_NODE_IMPORTANCE, MODEL_TYPE_TO_GNN_LAYER_NAME
from ..base_explainer import BaseExplainer
from ..explanation import GraphExplanation
from histocartography.utils.torch import torch_to_list, torch_to_numpy


class AttentionGNNExplainer(BaseExplainer):
    def __init__(
            self,
            model,
            config, 
            cuda=False,
            verbose=False
    ):
        """
        Attention-based method for GNN explanation constructor 
        :param model: (nn.Module) a pre-trained model to run the forward pass 
        :param config: (dict) method-specific parameters 
        :param cuda: (bool) if cuda is enable 
        :param verbose: (bool) if verbose is enable
        """
        super(AttentionGNNExplainer, self).__init__(model, config, cuda, verbose)
        self.gnn_layer_name = MODEL_TYPE_TO_GNN_LAYER_NAME[config['model_params']['model_type']]

    def explain(self, data, label):
        """
        Explain a graph instance
        :param data: (DGLGraph?) graph
        :param label: (int) label for the input data 
        """

        # 1/ pre-processing
        graph = deepcopy(data[0])
        image = data[1]
        image_name = data[2]
        self.model.eval()
        if self.cuda:
            self.model = self.model.cuda()
        self.model.zero_grad()
        self.model.set_forward_hook(self.model.pred_layer.mlp, '0')  # hook before the last classification layer

        # 2/ forward-pass and attention
        logits = self.model(data)
        attention_weights = [getattr(self.model, self.gnn_layer_name).layers[j].heads[i].attn_weights for j in range(len(getattr(self.model, self.gnn_layer_name).layers)) for i in range(len(getattr(self.model, self.gnn_layer_name).layers[0].heads))]
        attention_weights = torch.sum(torch.stack(attention_weights, dim=0), dim=0)
        # norm_attention_weights = self.model.cell_graph_gnn.layers[-1].heads[0].norm_attn_weights
        node_importance = self._compute_node_importance(attention_weights, graph).squeeze()
        # norm_node_importance = self._compute_node_importance(norm_attention_weights, graph)
        graph.ndata['node_importance'] = node_importance

        # 3/ prune the graph at different thresholds using the node importance -- then forward again and store information
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

    def _compute_node_importance(self, attention_weights, graph):

        def msg_func(edges):
            return {'a': attention_weights}

        def reduce_func(nodes):
            node_importance = torch.sum(nodes.mailbox['a'], dim=1)
            return {'node_importance': node_importance}

        graph.update_all(msg_func, reduce_func)
        return graph.ndata.pop('node_importance')






