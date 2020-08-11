import torch

from histocartography.utils.io import get_device
from ..base_explainer import BaseExplainer


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

        # @TODO: implementation for LRP 

    def explain(self, data, label):
        """
        Explain a graph instance
        :param data: (?) graph/image/tuple
        :param label: (int) label for the input data 
        """
        raise NotImplementedError('Needs to be implemented.')
