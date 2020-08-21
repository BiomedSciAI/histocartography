import torch

from histocartography.utils.io import get_device
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

    def explain(self, data, label):
        """
        Explain a graph instance
        :param data: (?) graph/image/tuple
        :param label: (int) label for the input data 
        """

        # 1/ pre-processing
        graph = data[0]
        image = data[1]
        image_name = data[2]
        model.eval()
        if self.cuda:
            self.model = self.model.cuda()

        # 2/ enable RLP 
        self.model.set_rlp(True)

        # 3/ forward pass 
        logits = self.model(data).squeeze()

        # 4/ extract R(t=T), ie last layer 
        max_idx = logits.argmax(dim=0)
        init_relevance = torch.zeros_like(logits)
        init_relevance[max_idx] = logits[max_idx]

        # 5/ apply iterative layer-wise relevance propagation 
        node_importance = self.model.rlp(init_relevance)
        graph.ndata['node_importance'] = torch.sum(node_importance, dim=1)

        # 6/ build and return explanation 
        explanation = GraphExplanation(
            self.config,
            image,
            image_name,
            logits,
            label,
            graph,
        )

        return explanation