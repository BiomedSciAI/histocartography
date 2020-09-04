# from histocartography.interpretability.base_explainer import BaseExplainer
# from histocartography.dataloader.constants import get_label_to_tumor_type

from base_explainer import BaseExplainer

from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
from torchcam.cams import CAM
from matplotlib import cm
from matplotlib import pyplot as plt


class CAMGNNExplainer(BaseExplainer):
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
        super(CAMGNNExplainer, self).__init__(model, config, cuda, verbose)

        self.savefig = './dcis.png'
        self.colormap = 'jet'

        # Set based on our trained CNN-single stream (10x)-ResNet34 network
        self.input_layer = '0'  # input_layer (str): name of the first layer
        self.conv_layer = '7'  # conv_layer (str): name of the last convolutional layer
        self.fc_layer = '11'  # fc_layer (str): name of the fully convolutional layer

        self.patch_size = 112
        self.patch_scale = 224

    def data_transformation(self, pil_img):
        img_tensor = normalize(to_tensor(resize(pil_img, (self.patch_scale, self.patch_scale))),
                               [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device=self.device)
        return img_tensor

    def get_patches(self, img):
        (h, w, c) = img.shape
        img = np.pad(img, ((0, self.patch_size), (0, self.patch_size),
                           (0, 0)), mode='constant', constant_values=255)

        self.patches = []
        x = 0
        while (x + self.patch_size) <= w:
            y = 0
            while (y + self.patch_size) <= h:
                patch = img[y: y + self.patch_size,
                        x: x + self.patch_size, :]
                self.patches.append(self.data_transformation(Image.fromarray(patch)))
        self.patches = torch.stack(self.patches)

    def explain(self, data, label):
        """
        Explain a image patch instance
        :param data: image (troi)
        :param label: (int) label for the input data
        """

        if self.cuda:
            self.model = self.model.cuda()

        self.model.eval()

        # Hook the corresponding layer in the model
        extractor = CAM(self.model, self.conv_layer, self.fc_layer)

        # Preprocess image
        img_tensor = self.data_transformation(data)

        # Get the class index
        self.model.zero_grad()
        scores = self.model(img_tensor.unsqueeze(0))
        #label = scores.squeeze(0).argmax().item() if label is None else label

        # Use the hooked data to compute activation map
        activation_map = extractor(label, scores).cpu()

        # Clean data
        extractor.clear_hooks()

        # Convert it to PIL image
        # The indexing below means first image in batch
        heatmap = to_pil_image(activation_map, mode='F')

        # Saliency map
        cmap = cm.get_cmap(self.colormap)
        heatmap = heatmap.resize(data.size, resample=Image.BICUBIC)
        heatmap = (255 * cmap(np.asarray(heatmap) ** 2)[:, :, 1:]).astype(np.uint8)

        return heatmap
























