import os
import torch
from torch.nn import Module
from abc import abstractmethod
from ..layers.multi_layer_gnn import MultiLayerGNN
from .zoo import MODEL_NAME_TO_URL, MODEL_NAME_TO_CONFIG
from ...utils import download_box_link


def get_number_of_classes(class_split):
    return len(class_split.split('VS'))


class BaseModel(Module):

    def __init__(
        self,
        class_split: str = None,
        num_classes: int = None,
        pretrained: bool = False
    ) -> None:
        """
        Base model constructor.

        Args:
            class_split (str): Class split. For instance in the BRACS dataset, one can specify
                               a 3-class split as: "benign+pathologicalbenign+udhVSadh+feaVSdcis+malignant".
                               Defaults to None.
            num_classes (int): Number of classes. Used if class split is not provided. Defaults to None.
            pretrained (bool): If loading pretrained checkpoint. Currently all the pretrained were trained on the BRACS dataset.
                               Defaults to False.
        """
        super().__init__()

        assert not(
            class_split is None and num_classes is None), "Please provide number of classes or class split."

        if class_split is not None:
            self.num_classes = get_number_of_classes(class_split)
        elif num_classes is not None:
            self.num_classes = num_classes
        else:
            raise ValueError(
                'Please provide either class split or number of classes. Not both.')

        self.pretrained = pretrained

    def _build_classification_params(self):
        """
        Build classification parameters
        """
        raise NotImplementedError('Implementation in subclasses.')

    def _load_checkpoint(self, model_name):
        checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            '..',
            'checkpoints'
        )
        download_box_link(
            url=MODEL_NAME_TO_URL[model_name],
            out_fname=os.path.join(checkpoint_path, model_name)
        )
        self.load_state_dict(
            torch.load(os.path.join(checkpoint_path, model_name))
        )

    @abstractmethod
    def forward(self, graph):
        """
        Forward pass
        """

    def set_forward_hook(self, module, layer):
        module._modules.get(layer).register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        """Activation hook"""
        self.latent_representation = output.data
