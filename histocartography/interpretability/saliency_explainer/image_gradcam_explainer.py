from ..base_explainer import BaseExplainer

from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.transforms.functional import resize, to_tensor, to_pil_image
from matplotlib import cm
import cv2

from histocartography.interpretability.saliency_explainer.grad_cam import LegacyGradCAM
from histocartography.interpretability.explanation import ImageExplanation
from histocartography.utils.torch import torch_to_list, torch_to_numpy
from histocartography.interpretability.constants import PATCH_SIZE, PATCH_SCALE, STRIDE, PATCH_RESIZE, data_transformation


class ImageGradCAMExplainer(BaseExplainer):
    def __init__(
            self,
            model,
            config,
            cuda=False,
            verbose=False
    ):
        """
        CAM for CNN-based saliency explanation constructor
        :param model: (nn.Module) a pre-trained model to run the forward pass
        :param config: (dict) method-specific parameters
        :param cuda: (bool) if cuda is enable
        :param verbose: (bool) if verbose is enable
        """
        super(ImageGradCAMExplainer, self).__init__(model, config, cuda, verbose)

        # Set based on our trained CNN-single stream (10x)-ResNet50 network
        self.conv_layer = '7'       # conv_layer (str): name of the last convolutional layer

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
        self.model.eval()

        # Preprocess image
        (h, w, c) = data.shape
        data = np.pad(data, ((0, PATCH_SIZE), (0, PATCH_SIZE),
                           (0, 0)), mode='constant', constant_values=255)

        # Initialize saliency map
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

                # Hook the corresponding layer in the model
                extractor = LegacyGradCAM(self.model, self.conv_layer)

                # Preprocess image
                patch_pil = data_transformation(Image.fromarray(patch), self.device)
                scores = self.model(patch_pil.unsqueeze(0))

                # Use the hooked data to compute activation map
                activation_map = extractor(label, scores).cpu()

                print('Activation map shape:', activation_map.shape)

                # Clean data
                extractor.clear_hooks()
                self.model.zero_grad()

                # Saliency map
                heatmap = to_pil_image(activation_map, mode='F')
                heatmap = heatmap.resize((PATCH_SIZE, PATCH_SIZE), resample=Image.BICUBIC)

                saliency_map[y: y + PATCH_SIZE, x: x + PATCH_SIZE] = np.asarray(heatmap)

                y += STRIDE

            x+= STRIDE

        saliency_map = saliency_map[:h, :w]

        print('#patches=', count)

        # convert heatmap to PIL image:
        saliency_map = Image.fromarray(saliency_map, mode='F')

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
