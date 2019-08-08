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
from scipy.stats import mode
#from PIL import Image, ImageDraw

# setup logging
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::IO::WSI')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
h1.setFormatter(formatter)
log.addHandler(h1)

levels_dict = {
    1 : '40x',
    2 : '20x',
    4 : '10x',
    8 : '5x',
    16 : '2.5x',
    32 : '1x',
    64 : '0.5x',
    128 : '0.25x',
    256 : '0.125x',
    512 : '0.625',
    1024 : '0.375'
}


# mapping of the magnification property used by the vendor
# usage:
# if Stack.properties['openslide.vendor'] not in ['phillips', 'generic-tiff']:
    # magnification = Stack.properties[magn_tag[Stack.properties['openslide.vendor']]]
    # magnification = Stack.properties['openslide.objective-power']
# Stack.properties[magn_tag[Stack.properties['openslide.vendor']]]
# 
magn_tag = {
    'aperio': 'aperio.AppMag', # maps to openslide.objective-power
    'hamamatsu': 'hamamatsu.SourceLens', # maps to openslide.objective-power
    'leica': 'leica.objective', # maps to openslide.objective-power
    'mirax': 'mirax.GENERAL.OBJECTIVE_MAGNIFICATION', # maps to openslide.objective-power
    'phillips': '', # NO MAPPING TO objective-power. Have to use mpp-x and mpp-y values
    'sakura': 'sakura.NominalLensMagnification', # maps to openslide.objective-power
    'trestle': 'trestle.Objective Power', # OFTEN INCORRECT. maps to openslide.objective-power
    'ventana': 'ventana.Magnification', # maps to openslide.objective-power
    'generic-tiff': '' # NO MAPPING TO objective-power.
}

def load(wsi_file=None, desired_level='10x'):
    """Loads a Whole Slide Image file at the desired magnification level

    Parameters
    ----------
    wsi_file : str
        The file containing the slide
    desired_level : str, default is '10x'
        Desired magnification level {5x, 10x, 20x, 40x}

    Returns
    -------
    Numpy Array image
        The WSI at the desired level as a Numpy array
    Double scale_factor
        The scale factor applied to extract the desired magnification level
    """

    log.debug('wsi_file : {}'.format(wsi_file))
    log.debug(os.path.isfile(wsi_file))

    Stack = open_slide(wsi_file)

    # magnification = Stack.properties['openslide.objective-power']
    if Stack.properties['openslide.vendor'] not in ['phillips', 'generic-tiff']:
        magnification = Stack.properties[magn_tag[Stack.properties['openslide.vendor']]]
    else:
        magnification = None
    log.debug('Original magnification: {}'.format(magnification))
    log.debug('Levels: {}'.format(Stack.level_count))
    log.debug('Level dimensions in stack: {}'.format(Stack.level_dimensions))

    downsamples = np.rint(np.asarray(Stack.level_downsamples)).astype(int)
    log.debug('Downsamples: {}'.format(downsamples))

    possible_resolutions  = [levels_dict.get(key) for key in downsamples]
    log.debug('Possible resolutions: {}'.format(possible_resolutions))
    possible_resolutions = np.asarray(possible_resolutions)

    level = np.where(possible_resolutions == desired_level)[0][0]
    log.debug('Level for desired resolution : {}'.format(level))

    size_0 = Stack.level_dimensions[0]
    size = Stack.level_dimensions[level]


    if (level <= 2):
        x = size[0]
        y = size[1]
        log.debug('Expected size of image: {},{}'.format(x, y))
        image = np.empty([y, x, 3], dtype=np.uint8)
        x_13_0 = int(size_0[0] / 3)
        y_13_0 = int(size_0[1] / 3)
        x_23_0 = 2 * x_13_0
        y_23_0 = 2 * y_13_0
        x_13 = int(size[0] / 3)
        y_13 = int(size[1] / 3)
        x_23 = 2 * x_13
        y_23 = 2 * y_13
        img_1 = np.asarray((Stack.read_region((0, 0), level, (x_13, y_13))).convert("RGB"))
        img_2 = np.asarray((Stack.read_region((x_13_0, 0), level, (x_13, y_13))).convert("RGB"))
        img_3 = np.asarray((Stack.read_region((x_23_0, 0), level, ((x - x_23), y_13))).convert("RGB"))

        img_4 = np.asarray((Stack.read_region((0, y_13_0), level, (x_13, y_13))).convert("RGB"))
        img_5 = np.asarray((Stack.read_region((x_13_0, y_13_0), level, (x_13, y_13))).convert("RGB"))
        img_6 = np.asarray((Stack.read_region((x_23_0, y_13_0), level, ((x - x_23), y_13))).convert("RGB"))

        img_7 = np.asarray((Stack.read_region((0, y_23_0), level, (x_13, (y - y_23)))).convert("RGB"))
        img_8 = np.asarray((Stack.read_region((x_13_0, y_23_0), level, (x_13, (y - y_23)))).convert("RGB"))
        img_9 = np.asarray((Stack.read_region((x_23_0, y_23_0), level, ((x - x_23), (y - y_23)))).convert("RGB"))

        image[0:y_13, 0:x_13, :] = img_1
        image[0:y_13, x_13:x_23, :] = img_2
        image[0:y_13, x_23:, :] = img_3

        image[y_13:y_23, 0:x_13, :] = img_4
        image[y_13:y_23, x_13:x_23, :] = img_5
        image[y_13:y_23, x_23:, :] = img_6

        image[y_23:, 0:x_13, :] = img_7
        image[y_23:, x_13:x_23, :] = img_8
        image[y_23:, x_23:, :] = img_9


    else:
        image = Stack.read_region((0, 0), level, (size[0], size[1]))
        image = image.convert("RGB")
        image = np.asarray(image)

    wSel, hSel = Stack.level_dimensions[level]
    wlower, hlower = Stack.level_dimensions[level+1]


    scale_factor = int(round(wSel / wlower))

    next_lower_resolution = possible_resolutions[level+1]
    log.debug('Next lower resolution: {}'.format(next_lower_resolution))
    log.debug('scale_factor: {}'.format(scale_factor))

    return image, next_lower_resolution, scale_factor
    

def save(file=None):
    """TODO. Currently returns method name"""
    log.debug("Saving file: {}".format(file))
    return 'Save'

def patch(file=None):
    """TODO. Currently returns method name"""
    log.debug("Getting patch from file: {}".format(file))
    return 'Patch'


