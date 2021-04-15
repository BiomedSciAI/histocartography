"""
Example: Stain normalize with Vahadane algorithm a list of H&E images.

Paper: "Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images", Vahadane et al, 2016.
"""

import os
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

from histocartography.preprocessing import VahadaneStainNormalizer
from histocartography.utils import download_example_data


def normalize_images(image_path):
    """
    Process the images in image path dir. In this dummy example,
    we use the first image as target for estimating normalization
    params.
    """

    # 1. get image path
    image_fnames = glob(os.path.join(image_path, '*.png'))

    # 2. define stain normalizer. If no target target is provided,
    # defaults ones are used. Note: Macenko normalization can be
    # defined in a similar way.
    target_image = image_fnames.pop(0)  # use the 1st image as target
    normalizer = VahadaneStainNormalizer(target_path=target_image)

    # 3. normalize all the images
    for image_path in tqdm(image_fnames):

        # a. load image
        _, image_name = os.path.split(image_path)
        image = np.array(Image.open(image_path))

        # b. apply Vahadane stain normalization
        norm_image = normalizer.process(image)

        # c. save the normalized image
        norm_image = Image.fromarray(np.uint8(norm_image))
        norm_image.save(
            os.path.join(
                'output',
                'normalized_images',
                image_name))


if __name__ == "__main__":

    # 1. download dummy images
    download_example_data('output')

    # 2. create output directory
    os.makedirs(os.path.join('output', 'normalized_images'), exist_ok=True)

    # 3. normalize images
    normalize_images(image_path=os.path.join('output', 'images'))
