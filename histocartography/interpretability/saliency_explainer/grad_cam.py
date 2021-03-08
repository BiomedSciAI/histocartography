from typing import List
import torch
import torch.nn.functional as F
import math 


EPS = 10e-7


class BaseCAM(object):

    def __init__(self, model: torch.nn.Module, conv_layers: List[str]) -> None:

        self.model = model
        self.hook_a = list()
        self.hook_g = list()
        self.hook_handles = list()

        # Forward hooks
        for conv_layer in conv_layers:
            if not hasattr(model, conv_layer):
                raise ValueError(f"Unable to find submodule {conv_layers} in the model")
            self.hook_handles.append(
                self.model._modules.get(conv_layer).register_forward_hook(self._hook_a)
            )
        # Backward hook
        for conv_layer in conv_layers:
            self.hook_handles.append(
                self.model._modules.get(conv_layer).register_backward_hook(self._hook_g)
            )
        # Enable hooks
        self._hooks_enabled = True
        # Should ReLU be used before normalization
        self._relu = True
        # Model output is used by the extractor
        self._score_used = False

    def _hook_a(self, module, input, output):
        """Hook activations (forward hook)"""
        if self._hooks_enabled:
            self.hook_a.append(output.data)

    def _hook_g(self, module, input, output):
        """Hook gradient (backward hook)"""
        if self._hooks_enabled:
            self.hook_g.append(output[0].data)

    def clear_hooks(self):
        """Clear model hooks"""
        for handle in self.hook_handles:
            handle.remove()

    @staticmethod
    def _normalize(cams):
        """CAM normalization"""
        cams -= cams.min(0).values
        cams /= cams.max(0).values + EPS
        return cams

    def _get_weights(self, class_idx, scores=None):
        raise NotImplementedError

    def _precheck(self, class_idx, scores):
        """Check for invalid computation cases"""

        # Check that forward has already occurred
        if self.hook_a is None:
            raise AssertionError(
                "Inputs need to be forwarded in the model for the conv features to be hooked"
            )

        # Check class_idx value
        if class_idx < 0:
            raise ValueError("Incorrect `class_idx` argument value")

        # Check scores arg
        if self._score_used and not isinstance(scores, torch.Tensor):
            raise ValueError(
                "model output scores is required to be passed to compute CAMs"
            )

    def __call__(self, class_idx, scores=None, normalized=True):
        """
        Compute the CAM for a specific output class

        Args:
            class_idx (int): output class index of the target class whose CAM will be computed
            scores (torch.Tensor[1, K], optional): forward output scores of the hooked model
            normalized (bool, optional): whether the CAM should be normalized

        Returns:
            torch.Tensor[M, N]: class activation map of hooked conv layer
        """

        # Integrity check
        self._precheck(class_idx, scores)

        # Get map weight
        weights = self._get_weights(class_idx, scores)
        is_cuda = weights.is_cuda

        # Perform the weighted combination to get the CAM
        forwards = torch.stack(self.hook_a, dim=2)
        num_nodes = forwards.squeeze(0).shape[0]
        batch_cams = (
            weights.unsqueeze(0).repeat(num_nodes, 1, 1) * forwards.squeeze(0)
        ).sum(dim=1)

        if is_cuda:
            batch_cams = batch_cams.cuda()

        if self._relu:
            batch_cams = F.relu(batch_cams, inplace=True)

        # Normalize the CAM
        if normalized:
            batch_cams = self._normalize(batch_cams)

        # Average out the different weights of the layers
        batch_cams = batch_cams.mean(dim=1)

        return batch_cams

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def _backprop(self, scores, class_idx):
        """Backpropagate the loss for a specific output class"""

        if self.hook_a is None:
            raise TypeError(
                "Apply forward path before calling backward hook."
            )

        # Backpropagate to get the gradients on the hooked layer
        loss = scores[:, class_idx].sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)


class GradCAM(BaseCAM):

    def __init__(self, model: torch.nn.Module, conv_layers: List[str]) -> None:
        """
        Class activation map extraction as in `"Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization" <https://arxiv.org/pdf/1610.02391.pdf>`_.
        Args:
            model (torch.nn.Module): input model
            conv_layer (str): name of the last convolutional layer
        """
        super().__init__(model, conv_layers)

    def _get_weights(self, class_idx, scores):
        """Computes the weight coefficients of the hooked activation maps"""
        # Backpropagate
        self._backprop(scores, class_idx)
        grads = torch.stack(list(reversed(self.hook_g)), dim=2)
        print('Grads', grads.shape)
        return grads.mean(axis=0)

    def __call__(self, *args, **kwargs):
        self.hook_g = list()
        return super().__call__(*args, **kwargs)


class GradCAMpp(BaseCAM):

    def __init__(self, model, conv_layer):
        """
        Class activation map extraction as in `"Grad-CAM++: Improved Visual Explanations for
        Deep Convolutional Networks" <https://arxiv.org/pdf/1710.11063.pdf>`_.

        Args:
            model (torch.nn.Module): input model
            conv_layer (str): name of the last convolutional layer
        """
        super().__init__(model, conv_layer)

    def _get_weights(self, class_idx, scores):
        """Computes the weight coefficients of the hooked activation maps"""

        # Backpropagate
        self._backprop(scores, class_idx)

        # Compute alpha 
        grad_2 = [f.pow(2) for f in self.hook_g]
        grad_3 = [f.pow(3) for f in self.hook_g]
        alpha = [g2 / (
            2 * g2 + (g3 * a).sum(axis=(0), keepdims=True) + EPS
        ) for g2, g3, a in zip(grad_2, grad_3, self.hook_a)
        ]

        weights = [a.squeeze_(0).mul_(torch.relu(g.squeeze(0))).sum(axis=(0)) for a, g in zip(alpha, self.hook_g)]
        weights = torch.stack(weights, dim=1)

        return weights

    def __call__(self, *args, **kwargs):
        self.hook_g = list()
        return super().__call__(*args, **kwargs)
