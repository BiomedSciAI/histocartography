"""
Example: Extract patch features on an image using a tissue mask.
"""

import os
from glob import glob
from PIL import Image
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from histocartography.preprocessing import MaskedGridDeepFeatureExtractor, GaussianTissueMask
from histocartography.utils import download_example_data


def masked_feature_extraction(image_path):
    """
    Extract patch features for all the images in image path dir and record (in)valid patches.
    """

    # 1. get image path
    image_fnames = glob(os.path.join(image_path, '*.png'))

    # 2. define feature extractor: extract features from ResNet50 after 'layer3'
    # on patches of size 256.
    extractor = MaskedGridDeepFeatureExtractor(architecture='resnet50',
                                               patch_size=256,
                                               tissue_thresh=0.1,
                                               downsample_factor=1,
                                               extraction_layer='layer3')

    # 3. process all the images
    for image_path in tqdm(image_fnames):

        # a. load image
        _, image_name = os.path.split(image_path)
        image_name = image_name.replace('.png', '')
        image = np.array(Image.open(image_path))

        # generate tissue mask
        mask_generator = GaussianTissueMask(sigma=5, kernel_size=15, downsampling_factor=4)
        mask = mask_generator.process(image=image)

        # b. extract index filter and patch features
        index_filter, features = extractor.process(image, mask)

        # c. reshape features, apply adaptive average pooling, and flatten again
        n_channels = 1024
        h = w = 16
        avg_pooler = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        feat_tensor = torch.tensor(np.array(features))
        feat_tensor = feat_tensor.reshape(feat_tensor.shape[-1], n_channels, h, w)
        feat_tensor = avg_pooler(feat_tensor).squeeze().cpu().detach().numpy()
        avg_features = pd.DataFrame(np.transpose(feat_tensor), columns=features.columns)

        # d. save all patch features
        np.save(os.path.join('output', 'masked_features', image_name), avg_features)


if __name__ == "__main__":

    # 1. download dummy images
    download_example_data('output')

    # 2. create output directory
    os.makedirs(os.path.join('output', 'masked_features'), exist_ok=True)

    # 3. normalize images
    masked_feature_extraction(image_path=os.path.join('output', 'images'))
