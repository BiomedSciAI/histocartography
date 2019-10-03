"""Whole Slide Image IO module."""
import logging
import sys
import os
import csv
import cv2
import itertools
import numpy as np
from scipy.sparse import coo_matrix

from lxml import etree
from openslide import open_slide

from .annotations import XMLAnnotation

# from PIL import Image, ImageDraw

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::IO::WSI')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
h1.setFormatter(formatter)
log.addHandler(h1)

SAFE_VENDORS = ['aperio', 'hamamatsu', 'leica', 'mirax', 'sakura', 'ventana']
# mapping of the mag property used by the vendor
# usage:
# if self.stack.properties['openslide.vendor'] in self.SAFE_VENDORS :
# mag = self.stack.properties[
#  self.MAGNIFICATION_TAG[
#  self.stack.properties['openslide.vendor']]]
# mag = self.stack.properties['openslide.objective-power']
# self.stack.properties[magn_tag[self.stack.properties['openslide.vendor']]]
#

MAGNIFICATION_TAG = {
    'aperio': 'aperio.AppMag',  # maps to openslide.objective-power
    'hamamatsu': 'hamamatsu.SourceLens',  # maps to openslide.objective-power
    'leica': 'leica.objective',  # maps to openslide.objective-power
    # maps to openslide.objective-power
    'mirax': 'mirax.GENERAL.OBJECTIVE_MAGNIFICATION',
    'phillips':
        '',  # NO MAPPING TO objective-power. Have to use mpp-x and mpp-y value
    'sakura':
        'sakura.NominalLensMagnification',  # maps to openslide.objective-power
    # OFTEN INCORRECT. maps to openslide.objective-power
    'trestle': 'trestle.Objective Power',
    'ventana': 'ventana.Magnification',  # maps to openslide.objective-power
    'generic-tiff': ''  # NO MAPPING TO objective-power.
}

DEFAULT_LABELS = [
    'background', 'NROI', '3+3', '3+4', '4+3', '4+4', '4+5', '5+5'
]


class WSI:
    """
    Whole Slide Image Class
    """

    def __init__(
        self, wsi_file, annotations=None, minimum_tissue_content=0.25
    ):
        """Constructs a WSI object with a given wsi_file

        Parameters
        ----------
        wsi_file : str
            The file containing the slide
        annotation_file : str, optional
            annotations for the file
        """

        self.wsi_file = wsi_file
        self.current_image = None
        self.current_mag = None
        self.current_downsample = None
        self.minimum_tissue_content = minimum_tissue_content
        self.annotations = annotations

        log.debug('wsi_file : %s', self.wsi_file)

        if os.path.isfile(self.wsi_file):
            self.stack = open_slide(self.wsi_file)
            properties = self.stack.properties
            self.vendor = properties.get('openslide.vendor', 'NONE')
        else:
            log.error('File does not exist')

        if self.vendor in SAFE_VENDORS:
            self.vendor_mag = properties[MAGNIFICATION_TAG[self.vendor]]
            self.openslide_mag = properties['openslide.objective-power']
            self.mag = float(self.openslide_mag)
            self.downsamples = np.rint(self.stack.level_downsamples
                                       ).astype(int)
        else:
            self.mag = 1
            self.vendor_mag = 1
            self.downsamples = np.asarray([1])
        self.available_mags = [
            self.mag / np.rint(downsample) for downsample in self.downsamples
        ]

        log.debug('Original mag: %s', self.vendor_mag)
        # log.debug('Original mag (openslide): %s', self.openslide_mag)
        log.debug('Levels: %s', self.stack.level_count)
        log.debug('Level dimensions %s', self.stack.level_dimensions)
        # log.debug('Downsamples: %s', self.downsamples)
        # log.debug('Possible resolutions: %s', self.available_mags)

    def image_at(self, mag=5):
        """gets the image at a desired mag level
        """

        # log.debug('Downsample for desired resolution : %s', self.mag / mag)
        level = self.stack.get_best_level_for_downsample(self.mag / mag)

        log.debug('Level for desired resolution : %s', level)

        size_0 = self.stack.level_dimensions[0]
        size = self.stack.level_dimensions[level]

        if level <= 2:
            full_width = size[0]
            full_height = size[1]
            default_width = int(size[0] / 3)
            default_height = int(size[1] / 3)
            # log.debug('Expected size of image: %s %s', full_height,
            #           full_height)
            image = np.empty([full_height, full_width, 3], dtype=np.uint8)

            for col in range(3):
                for row in range(3):
                    original_x_pos = int(size_0[0] * (col) / 3)
                    original_y_pos = int(size_0[1] * (row) / 3)

                    if col < 2:
                        target_width = default_width
                    else:
                        target_width = full_width - 2 * default_width
                    if row < 2:
                        target_height = default_height
                    else:
                        target_height = full_height - 2 * default_height
                    region = self.stack.read_region(
                        (original_x_pos, original_y_pos), level,
                        (target_width, target_height)
                    )

                    region = np.asarray(region.convert("RGB"))
                    row_pos = row * default_height
                    col_pos = col * default_width
                    image[row_pos:row_pos + target_height, col_pos:col_pos +
                          target_width] = region

        else:
            image = self.stack.read_region((0, 0), level, (size[0], size[1]))
            image = image.convert("RGB")
            image = np.asarray(image)

        log.debug('Image shape after loading %s', image.shape)

        self.current_image = image
        self.current_mag = mag
        self.current_downsample = self.mag / mag

        return image

    def tissue_threshold(self):

        image = self.image_at(self.mag / 32)
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img_inv = (255 - img_gray)  # invert the image intensity
        threshold, mask_ = cv2.threshold(
            img_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return threshold

    def tissue_mask_at(self, mag=5):
        """gets the tissue mask at a desired mag level
        """

        if self.current_mag == mag and self.current_image is not None:
            image = self.current_image
        else:
            image = self.image_at(mag)

        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img_inv = (255 - img_gray)  # invert the image intensity
        _, mask_ = cv2.threshold(
            img_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        result = cv2.findContours(
            mask_, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(result) == 2:
            contour = result[0]
        elif len(result) == 3:
            contour = result[1]

        for cnt in contour:
            cv2.drawContours(mask_, [cnt], 0, 255, -1)

        # --- removing small connected components ---
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
            mask_, connectivity=8
        )
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        mask_remove_small = np.zeros((output.shape))
        remove_blob_size = 5000

        for i in range(0, nb_components):
            if sizes[i] >= remove_blob_size:
                mask_remove_small[output == i + 1] = 255

        mask_remove_small = mask_remove_small.astype(int)
        mask_remove_small = np.uint8(mask_remove_small)

        mask = np.zeros((mask_.shape[0], mask_.shape[1]), np.uint8)
        mask[mask_remove_small == 255] = 255  # NROI

        log.debug('tissue mask generated')

        return mask

    def patches(
        self,
        origin=(0, 0),
        size=(128, 128),
        stride=(128, 128),
        mag=5,
        shuffle=False,
        annotations=False
    ):
        """
        Patches generator. It initializes with shape and stride for a given
        magnification, and will produce new patches as it is called with
        next()
        origin (tuple): Origin of the patch extraction. Defaults to 0,0.
        size(tuple): size of the patches. Defaults to 128 ,128
        stride(tuple): stride to extract patches consecutively
            Defaults to 128 ,128
        mag (float): magnification at which the patches will be extracted.
            Defaults to 5 (5x)
        shuffle (bool): whether to shuffle patches before start the generator
        annotations(bool): whether to compute the annotations or not.
            If False (default) a patch of zeroes will be returned

        """

        full_width = self.stack.level_dimensions[0][0]
        full_height = self.stack.level_dimensions[0][1]

        downsample, level, xy_positions = self.patch_positions(
            origin, size, stride, mag
        )

        tissue_threshold = self.tissue_threshold()

        for x, y in xy_positions:

            x_mag = int(x / downsample)
            y_mag = int(y / downsample)

            region, patch_labels = self.get_patch_with_labels(
                downsample, level, (x, y), size
            )
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            tissue_pixels = np.sum(np.where(gray < tissue_threshold))
            # log.debug("Region %s,%s has %s pixels ratio", x, y,
            #           tissue_pixels / region.size)
            if tissue_pixels > self.minimum_tissue_content:
                yield (
                    x, y, full_width, full_height, x_mag, y_mag, region,
                    patch_labels
                )

    def get_patch_with_labels(self, downsample, level, xy_position, size):
        x, y = xy_position
        patch_labels = np.zeros(size, dtype=np.uint8)
        if self.annotations is not None:
            patch_labels = self.annotations.mask(size, (x, y), downsample)
        try:
            region = np.array(self.stack.read_region((x, y), level, size))
        except OSError:
            log.warning(
                f'Patch error at {x},{y} of size {size} from {self.wsi_file}'
            )
            region = np.zeros((size[0], size[1], 3), dtype=np.uint8)

        patch_labels = self.annotations.mask(size, (x, y), downsample)
        return region, patch_labels

    def patch_positions(
        self, origin=(0, 0), size=(128, 128), stride=(128, 128), mag=5
    ):

        full_width = self.stack.level_dimensions[0][0]
        full_height = self.stack.level_dimensions[0][1]
        downsample = self.mag / mag
        level = self.stack.get_best_level_for_downsample(downsample)
        horiz_step = int(stride[0] * downsample)
        vert_step = int(stride[1] * downsample)
        x_positions = np.arange(origin[0], full_height, horiz_step)
        y_positions = np.arange(origin[1], full_width, vert_step)

        log.debug('Level for desired resolution : %s', level)
        log.debug('Step size : %s %s', horiz_step, vert_step)
        log.debug('Num Patches : %s %s', len(x_positions), len(y_positions))

        xy_positions = list(itertools.product(x_positions, y_positions))

        return downsample, level, xy_positions
