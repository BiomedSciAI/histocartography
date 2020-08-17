import torch

from histocartography.utils.io import get_device
from ..base_explainer import BaseExplainer
from ..explanation import GraphExplanation


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

    def explain(self, data, label):
        """
        Explain a graph instance
        :param data: (DGLGraph?) graph
        :param label: (int) label for the input data 
        """

        graph = data[0]
        image = data[1]
        image_name = data[2]

        if self.cuda:
            self.model = self.model.cuda()

        logits = self.model(data)
        attention_weights = [self.model.cell_graph_gnn.layers[j].heads[i].attn_weights for j in range(len(self.model.cell_graph_gnn.layers)) for i in range(len(self.model.cell_graph_gnn.layers[0].heads))]
        attention_weights = torch.sum(torch.stack(attention_weights, dim=0), dim=0)
        # norm_attention_weights = self.model.cell_graph_gnn.layers[-1].heads[0].norm_attn_weights

        node_importance = self._compute_node_importance(attention_weights, graph)
        # norm_node_importance = self._compute_node_importance(norm_attention_weights, graph)

        graph.ndata['node_importance'] = node_importance

        # build explanation object
        explanation = GraphExplanation(
            self.config,
            image,
            image_name,
            logits,
            label,
            graph,
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






