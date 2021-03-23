from torch.nn import Module

from histocartography.ml.layers.multi_layer_gnn import MultiLayerGNN
from histocartography.dataloader.constants import get_number_of_classes


class BaseModel(Module):

    def __init__(
        self,
        class_split: str = None,
        num_classes: int = None
        ) -> None:
        """
        Base model constructor.

        Args:
            class_split (str): Class split. For instance in the BRACS dataset, one can specify
                               a 3-class split as: "benign+pathologicalbenign+udhVSadh+feaVSdcis+malignant".
                               Default to None. 
            num_classes (int): Number of classes. Used if class split is not provided. Default to None. 
        """
        super().__init__()

        assert not(class_split is None and num_classes is None), "Please provide number of classes or class split."

        if class_split is not None:
            self.num_classes = get_number_of_classes(class_split)
        elif num_classes is not None:
            self.num_classes = num_classes
        else:
            raise ValueError('Please provide either class split or number of classes. Not both.')

    def _build_classification_params(self):
        """
        Build classification parameters
        """
        raise NotImplementedError('Implementation in subclasses.')

    def forward(self, graphs):
        """
        Forward pass
        :param graphs:
        """
        raise NotImplementedError('Implementation in subclasses.')

    def set_forward_hook(self, module, layer):
        module._modules.get(layer).register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        """Activation hook"""
        self.latent_representation = output.data
