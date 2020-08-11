import torch

from histocartography.utils.io import get_device


class BaseExplainer:
    def __init__(
            self,
            model,
            config, 
            cuda=False,
            verbose=False
    ):
        """
        Base Explainer constructor 
        :param model: (nn.Module) a pre-trained model to run the forward pass 
        :param config: (dict) method-specific parameters 
        :param cuda: (bool) if cuda is enable 
        :param verbose: (bool) if verbose is enable
        """
        self.model = model
        self.config = config
        self.cuda = cuda
        self.device = get_device(self.cuda)
        self.verbose = verbose

    def explain(self, data, label):
        """
        Explain a graph instance
        :param data: (?) graph/image/tuple
        :param label: (int) label for the input data 
        """
        raise NotImplementedError('Implementation in sub classes.')