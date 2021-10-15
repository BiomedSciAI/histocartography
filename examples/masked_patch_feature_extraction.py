"""
Example: Extract patch features on an image using a tissue mask.
"""

import os
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

from histocartography.preprocessing import MaskedGridDeepFeatureExtractor, GaussianTissueMask
from histocartography.utils import download_example_data


def masked_feature_extraction(image_path):
    """
    Extract patch features for all the images in image path dir and record (in)valid patches.
    """

    # 1. get image path
    image_fnames = glob(os.path.join(image_path, '*.png'))

    # 2. define feature extractor: extract features on patches of size 224
    # and set the minimum fraction of tissue for a patch to be considered as valid.
    extractor = MaskedGridDeepFeatureExtractor(architecture='resnet34',
                                               patch_size=224,
                                               tissue_thresh=0.1,
                                               downsample_factor=2)

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

        # c. save all patch features
        np.save(os.path.join('output', 'masked_features', image_name), features)


if __name__ == "__main__":

    # 1. download dummy images
    download_example_data('output')

    # 2. create output directory
    os.makedirs(os.path.join('output', 'masked_features'), exist_ok=True)

    # 3. normalize images
    masked_feature_extraction(image_path=os.path.join('output', 'images'))
