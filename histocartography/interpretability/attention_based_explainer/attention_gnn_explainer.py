import torch

from histocartography.utils.io import get_device
from ..base_explainer import BaseExplainer


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
        attention_weights = self.model.attn_weights
        norm_attention_weights = self.model.norm_attn_weights
        num_nodes = self.

        node_importance = self._compute_node_importance(attention_weights, graph.edges())
        norm_node_importance = self._compute_node_importance(norm_attention_weights, graph.edges())

        graph.ndata['node_importance'] = node_importance

        # build explanation object
        explanation = GraphExplanation(
            self.config,
            self.image,
            self.image_name,
            self.logits,
            label,
            graph,
        )

    def _compute_node_importance(self, attention_weights, graph):

        # debug purposes 
        print('Attention weights:', attention_weights.shape)

        def msg_func(edges):
            return {'a': attention_weights}

        def reduce_func(nodes):
            node_importance = nodes.mailbox['a']
            return {'node_importance': node_importance}

        graph.update_all(message_func, reduce_func)
        return graph.ndata.pop('node_importance')
