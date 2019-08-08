"""Whole Slide Image IO module."""
import logging
import sys

import numpy as np
from lxml import etree
import glob, os
import csv
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import cv2
from PIL import Image

from histocartography.io.wsi import load


# from PIL import Image, ImageDraw

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::PREPROCESING::TISSUE_MASK')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
h1.setFormatter(formatter)
log.addHandler(h1)


def get_tissue_mask(image=None):

    """For generating mask to get only tissue content from the image

    Parameters
    ----------
    image : numpy array
        The image loaded in numpy array

    Returns
    -------
    Numpy Array mask
        The tissue mask of the input image, dimensions same as that of image
    """

    remove_blob_size = 5000

    # ---- Create mask for the image to get only tissue content

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_inv = (255 - img_gray)  # invert the image intensity
    val_thr_stained, mask_ = cv2.threshold(img_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    _, contour, _ = cv2.findContours(mask_, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(mask_, [cnt], 0, 255, -1)

    # --- removing small connected components ---
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # minimum size of particles we want to keep (number of pixels)

    mask_remove_small = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above remove_blob_size
    for i in range(0, nb_components):
        if sizes[i] >= remove_blob_size:
            mask_remove_small[output == i + 1] = 255

    mask_remove_small = mask_remove_small.astype(int)
    mask_remove_small = np.uint8(mask_remove_small)

    mask = np.zeros((mask_.shape[0], mask_.shape[1]), np.uint8)
    mask[mask_remove_small == 255] = 255  # NROI

    log.debug('tissue mask generated')


    return mask