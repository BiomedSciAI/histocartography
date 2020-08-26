import torch
from copy import deepcopy

from histocartography.utils.io import get_device
from histocartography.interpretability.constants import KEEP_PERCENTAGE_OF_NODE_IMPORTANCE
from ..base_explainer import BaseExplainer
from ..explanation import GraphExplanation


class LRPGNNExplainer(BaseExplainer):
    def __init__(
            self,
            model,
            config, 
            cuda=False,
            verbose=False
    ):
        """
        LRP for GNN explanation constructor 
        :param model: (nn.Module) a pre-trained model to run the forward pass 
        :param config: (dict) method-specific parameters 
        :param cuda: (bool) if cuda is enable 
        :param verbose: (bool) if verbose is enable
        """
        super(LRPGNNExplainer, self).__init__(model, config, cuda, verbose)

    def _apply_rlp(self, data):
        logits = self.model([deepcopy(data[0])]).squeeze()
        max_idx = logits.argmax(dim=0)
        init_relevance = torch.zeros_like(logits)
        init_relevance[max_idx] = logits[max_idx]
        node_importance = self.model.rlp(init_relevance)
        node_importance = torch.sum(node_importance, dim=1)
        return node_importance

    def explain(self, data, label):
        """
        Explain a graph instance
        :param data: (?) graph/image/tuple
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
        self.model.set_rlp(True)

        # 3/ forward pass and RLP
        node_importance = self._apply_rlp(data)
        graph.ndata['node_importance'] = node_importance

        # 4/ prune the graph at different thresholds using the node importance -- then forward again and store information
        explanation_graphs = {}
        for keep_percentage in KEEP_PERCENTAGE_OF_NODE_IMPORTANCE:
            # a. prune graph
            pruned_graph = self._build_pruned_graph(graph, keep_percentage)
            # b. forward pass
            logits = self.model([pruned_graph])
            # c. store in dict 
            explanation_graphs[keep_percentage] = {}
            explanation_graphs[keep_percentage]['logits'] = logits.cpu().detach().numpy().tolist()
            explanation_graphs[keep_percentage]['latent'] = self.model.latent_representation.clone().cpu().detach().numpy().tolist()
            explanation_graphs[keep_percentage]['num_nodes'] = pruned_graph.number_of_nodes()
            explanation_graphs[keep_percentage]['num_edges'] = pruned_graph.number_of_edges()

        # 5/ build and return explanation 
        explanation = GraphExplanation(
            self.config,
            image,
            image_name,
            label,
            explanation_graphs,
        )

        return explanation