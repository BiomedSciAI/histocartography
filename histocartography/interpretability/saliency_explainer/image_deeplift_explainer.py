from ..base_explainer import BaseExplainer

from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
import cv2
from captum.attr import DeepLift

from histocartography.interpretability.explanation import ImageExplanation
from histocartography.utils.torch import torch_to_list, torch_to_numpy

PATCH_SIZE = 448
PATCH_SCALE = 224
STRIDE = 448
PATCH_RESIZE = 112


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

    def data_transformation(self, pil_img):
        img_tensor = normalize(to_tensor(resize(pil_img, (PATCH_SCALE, PATCH_SCALE))),
                               [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device=self.device)
        return img_tensor

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
        saliency_map = np.zeros_like(data)
        saliency_count = np.zeros_like(data)

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
                patch_pil = self.data_transformation(Image.fromarray(patch)).unsqueeze(0)
                patch_pil.requires_grad = True

                # Forward & feature attribution 
                scores = self.model(patch_pil)
                prediction = scores.argmax(dim=1)
                activation_map = self.attribute_image_features(self.eval_deep_lift, patch_pil, prediction, baselines=patch_pil * 0)
                activation_map = np.transpose(activation_map.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

                # # Saliency map
                activation_map = to_pil_image(activation_map, mode='RGB')
                activation_map = activation_map.resize((PATCH_SIZE, PATCH_SIZE), resample=Image.BICUBIC)

                saliency_map[y: y + PATCH_SIZE, x: x + PATCH_SIZE, :] += (255 * np.asarray(activation_map)).astype(np.uint8)
                saliency_count[y: y + PATCH_SIZE, x: x + PATCH_SIZE, :] += 1

                y += STRIDE

            x+= STRIDE

        # why doing this ?
        saliency_map = saliency_map[:h, :w, :]  
        saliency_count = saliency_count[:h, :w, :]
        saliency_map = np.divide(saliency_map, saliency_count)

        print('#patches=', count)

        # Create ImageExplanation object
        explanation = ImageExplanation(
            self.config,
            image,
            image_name,
            label,
            saliency_map,
            torch_to_list(scores)
        )

        return explanation
























