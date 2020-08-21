#from histocartography.interpretability.base_explainer import BaseExplainer

from base_explainer import BaseExplainer

from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
from torchcam.cams import SmoothGradCAMpp
from matplotlib import cm
import cv2


class SmoothGradCAMPPGNNExplainer(BaseExplainer):
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
        super(SmoothGradCAMPPGNNExplainer, self).__init__(model, config, cuda, verbose)

        self.colormap='jet'

        # Set based on our trained CNN-single stream (10x)-ResNet34 network
        self.input_layer = '0'      # input_layer (str): name of the first layer
        self.conv_layer = '7'       # conv_layer (str): name of the last convolutional layer
        self.fc_layer = '11'        # fc_layer (str): name of the fully convolutional layer

        self.patch_size = 448
        self.patch_scale = 224
        self.stride = 448
        self.patch_resize = 112
        self.cmap = cm.get_cmap(self.colormap)


    def data_transformation(self, pil_img):
        img_tensor = normalize(to_tensor(resize(pil_img, (self.patch_scale, self.patch_scale))),
                               [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device=self.device)
        return img_tensor



    def explain(self, data, label):
        """
        Explain a image patch instance
        :param data: image (troi)
        :param label: (int) label for the input data
        """
        data = np.array(data)

        if self.cuda:
            self.model = self.model.cuda()
        self.model.eval()
        self.model.zero_grad()

        # Preprocess image
        (h, w, c) = data.shape
        data = np.pad(data, ((0, self.patch_size), (0, self.patch_size),
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

                patch = data[y: y + self.patch_size,
                             x: x + self.patch_size, :]
                patch = cv2.resize(patch, (self.patch_resize, self.patch_resize), interpolation=cv2.INTER_NEAREST)

                # Hook the corresponding layer in the model
                extractor = SmoothGradCAMpp(self.model, self.conv_layer, self.input_layer)

                # Preprocess image
                patch_pil = self.data_transformation(Image.fromarray(patch))

                scores = self.model(patch_pil.unsqueeze(0))

                # Use the hooked data to compute activation map
                activation_map = extractor(label, scores).cpu()

                # Clean data
                extractor.clear_hooks()

                # Saliency map
                heatmap = to_pil_image(activation_map, mode='F')
                heatmap = heatmap.resize((self.patch_size, self.patch_size), resample=Image.BICUBIC)

                saliency_map[y: y + self.patch_size, x: x + self.patch_size, :] += \
                    (255 * self.cmap(np.asarray(heatmap) ** 2)[:, :, 1:]).astype(np.uint8)

                saliency_count[y: y + self.patch_size, x: x + self.patch_size, :] += 1

                y += self.stride

            x+= self.stride

        saliency_map = saliency_map[:h, :w, :]
        saliency_count = saliency_count[:h, :w, :]
        saliency_map = np.divide(saliency_map, saliency_count)

        print('#patches=', count)
        return saliency_map
























