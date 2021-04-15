"""
Example: Extract a feature cube on an image.

As used in:
- "Neural Image Compression for Gigapixel Histopathology Image Analysis", Tellez et al, 2018.
- "Context-aware convolutional neural network for grading of colorectal cancer histology images", Shaban et al, 2020.
"""

import os
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

from histocartography.preprocessing import GridDeepFeatureExtractor
from histocartography.utils import download_example_data


def feature_cube_extraction(image_path):
    """
    Extract feature cube for all the images in image path dir.
    """

    # 1. get image path
    image_fnames = glob(os.path.join(image_path, '*.png'))

    # 2. define feature extractor: extract features on patches of size 224
    # without stride.
    extractor = GridDeepFeatureExtractor('resnet34', patch_size=224)

    # 3. process all the images
    for image_path in tqdm(image_fnames):

        # a. load image
        _, image_name = os.path.split(image_path)
        image_name = image_name.replace('.png', '')
        image = np.array(Image.open(image_path))

        # b. extract feature cube
        features = extractor.process(image)

        # c. save the features
        np.save(os.path.join('output', 'feature_cubes', image_name), features)


if __name__ == "__main__":

    # 1. download dummy images
    download_example_data('output')

    # 2. create output directory
    os.makedirs(os.path.join('output', 'feature_cubes'), exist_ok=True)

    # 3. normalize images
    feature_cube_extraction(image_path=os.path.join('output', 'images'))
