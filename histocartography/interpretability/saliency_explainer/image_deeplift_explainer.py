from ..base_explainer import BaseExplainer

from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.transforms.functional import resize, to_tensor, to_pil_image
import cv2
from captum.attr import DeepLift

from histocartography.interpretability.explanation import ImageExplanation
from histocartography.utils.torch import torch_to_list, torch_to_numpy
from histocartography.interpretability.constants import PATCH_SIZE, PATCH_SCALE, STRIDE, PATCH_RESIZE, data_transformation


class ImageDeepLiftExplainer(BaseExplainer):
    def __init__(
            self,
            model,
            config,
            cuda=False,
            verbose=False
    ):
        """
        DeepLift for Images constructor
        :param model: (nn.Module) a pre-trained model to run the forward pass
        :param config: (dict) method-specific parameters
        :param cuda: (bool) if cuda is enable
        :param verbose: (bool) if verbose is enable
        """
        super(ImageDeepLiftExplainer, self).__init__(model, config, cuda, verbose)
        self.eval_deep_lift = DeepLift(self.model)

    def attribute_image_features(self, algorithm, image, label, **kwargs):
        self.model.zero_grad()
        tensor_attributions = algorithm.attribute(image,
                                                  target=label,
                                                  **kwargs
                                                 )
        return tensor_attributions

    def explain(self, data, label):
        """
        Explain a image patch instance
        :param data: image (troi)
        :param label: (int) label for the input data
        """

        # 1. Extract data 
        image = data[0][0]
        image_name = data[-1][0]
        data = np.array(image)

        if self.cuda:
            self.model = self.model.cuda()

        # Preprocess image
        (h, w, c) = data.shape
        data = np.pad(data, ((0, PATCH_SIZE), (0, PATCH_SIZE),
                           (0, 0)), mode='constant', constant_values=255)

        # Initialize saliency map
        # saliency_map = np.zeros_like(data).astype(np.float)
        saliency_map = np.zeros_like(data[:, :, 0]).astype(np.float)

        # Patch-wise processing
        count = 0
        x = 0
        while x <= w:
            y = 0
            while y <= h:
                count += 1
                if count % 10 == 0:
                    print('Done: ', count)

                patch = data[y: y + PATCH_SIZE,
                             x: x + PATCH_SIZE, :]
                patch = cv2.resize(patch, (PATCH_RESIZE, PATCH_RESIZE), interpolation=cv2.INTER_NEAREST)

                # Preprocess image
                patch_tensor = data_transformation(Image.fromarray(patch), self.device).unsqueeze(0)
                patch_tensor.requires_grad = True

                # Forward & feature attribution 
                scores = self.model(patch_tensor)
                prediction = scores.argmax(dim=1)
                activation_map = self.attribute_image_features(self.eval_deep_lift, patch_tensor, prediction, baselines=patch_tensor * 0)
                # print('Activation map:', activation_map.shape)
                # activation_map = np.transpose(activation_map.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

                # # Saliency map
                # heatmap = to_pil_image(activation_map, mode='RGB')
                # heatmap = heatmap.resize((PATCH_SIZE, PATCH_SIZE), resample=Image.BICUBIC)
                # saliency_map[y: y + PATCH_SIZE, x: x + PATCH_SIZE, :] = np.asarray(heatmap) 

                # trying to sum over the color channels beforehand 
                activation_map = activation_map.sum(dim=1).squeeze(0).cpu().detach().numpy()

                # Saliency map
                heatmap = to_pil_image(activation_map, mode='F')
                heatmap = heatmap.resize((PATCH_SIZE, PATCH_SIZE), resample=Image.BICUBIC)
                saliency_map[y: y + PATCH_SIZE, x: x + PATCH_SIZE] = np.asarray(heatmap) 

                y += STRIDE

            x+= STRIDE

        # convert heatmap to PIL image
        # saliency_map = Image.fromarray(saliency_map[:h, :w, :], mode='RGB')
        saliency_map = Image.fromarray((255 * saliency_map[:h, :w]).astype(np.uint8))

        print('#patches=', count)

        # Create ImageExplanation object
        explanation = ImageExplanation(
            self.config,
            image,
            image_name,
            label,
            saliency_map,
            torch_to_list(scores.squeeze())
        )

        return explanation
























