"""Whole Slide Image IO module."""
import logging
import sys
import os
import csv

import cv2

import numpy as np

from lxml import etree
from openslide import open_slide

# from PIL import Image, ImageDraw

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
LOG = logging.getLogger('Histocartography::IO::WSI')
H1 = logging.StreamHandler(sys.stdout)
LOG.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
H1.setFormatter(FORMATTER)
LOG.addHandler(H1)


SAFE_VENDORS = ['aperio', 'hamamatsu', 'leica', 'mirax', 'sakura', 'ventana']
# mapping of the mag property used by the vendor
# usage:
# if self.stack.properties['openslide.vendor'] in self.SAFE_VENDORS :
# mag = self.stack.properties[
#                                       self.MAGNIFICATION_TAG[
#                                       self.stack.properties['openslide.vendor']]]
# mag = self.stack.properties['openslide.objective-power']
# self.stack.properties[magn_tag[self.stack.properties['openslide.vendor']]]
#

MAGNIFICATION_TAG = {
    'aperio': 'aperio.AppMag',  # maps to openslide.objective-power
    'hamamatsu': 'hamamatsu.SourceLens',  # maps to openslide.objective-power
    'leica': 'leica.objective',  # maps to openslide.objective-power
    # maps to openslide.objective-power
    'mirax': 'mirax.GENERAL.OBJECTIVE_MAGNIFICATION',
    'phillips': '',  # NO MAPPING TO objective-power. Have to use mpp-x and mpp-y values
    'sakura': 'sakura.NominalLensMagnification',  # maps to openslide.objective-power
    # OFTEN INCORRECT. maps to openslide.objective-power
    'trestle': 'trestle.Objective Power',
    'ventana': 'ventana.Magnification',  # maps to openslide.objective-power
    'generic-tiff': ''  # NO MAPPING TO objective-power.
}

DEFAULT_LABELS = np.array(
    ['background', 'NROI', '3+3', '3+4', '4+3', '4+4', '4+5', '5+5'])


class WSI:
    """
    Whole Slide Image Class
    """

    def __init__(self, wsi_file, annotation_file=None, annotation_labels=DEFAULT_LABELS):
        """Constructs a WSI object with a given wsi_file

        Parameters
        ----------
        wsi_file : str
            The file containing the slide
        annotation_file : str, optional
            annotations for the file
        """

        self.wsi_file = wsi_file
        self.annotation_file = annotation_file
        self.annotation_labels = annotation_labels
        self.current_image = None
        self.current_mag = None
        self.current_downsample = None

        LOG.debug('wsi_file : %s', self.wsi_file)

        if os.path.isfile(self.wsi_file):
            self.stack = open_slide(self.wsi_file)
            properties = self.stack.properties
            self.vendor = properties['openslide.vendor']
        else:
            LOG.error('File does not exist')

        if self.stack.properties['openslide.vendor'] in SAFE_VENDORS:
            self.vendor_mag = properties[MAGNIFICATION_TAG[self.vendor]]
            self.openslide_mag = properties['openslide.objective-power']
            self.mag = float(self.openslide_mag)
        else:
            self.mag = None
        self.downsamples = np.rint(self.stack.level_downsamples).astype(int)
        self.available_mags = [
            self.mag/np.rint(downsample) for downsample in self.downsamples]

        LOG.debug('Original mag: %s', self.vendor_mag)
        LOG.debug('Original mag (openslide): %s', self.openslide_mag)
        LOG.debug('Levels: %s', self.stack.level_count)
        LOG.debug('Level dimensions %s', self.stack.level_dimensions)
        LOG.debug('Downsamples: %s', self.downsamples)
        LOG.debug('Possible resolutions: %s', self.available_mags)

    def image_at(self, mag=5):
        """gets the image at a desired mag level
        """

        LOG.debug('Downsample for desired resolution : %s', self.mag / mag)
        level = self.stack.get_best_level_for_downsample(self.mag / mag)

        LOG.debug('Level for desired resolution : %s', level)

        size_0 = self.stack.level_dimensions[0]
        size = self.stack.level_dimensions[level]

        if level <= 2:
            full_width = size[0]
            full_height = size[1]
            default_width = int(size[0] / 3)
            default_height = int(size[1] / 3)
            LOG.debug('Expected size of image: %s %s',
                      full_height, full_height)
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
                        (original_x_pos, original_y_pos),
                        level,
                        (target_width, target_height)
                    )

                    region = np.asarray(region.convert("RGB"))
                    row_pos = row*default_height
                    col_pos = col*default_width
                    image[row_pos:row_pos + target_height,
                          col_pos:col_pos + target_width] = region

        else:
            image = self.stack.read_region((0, 0), level, (size[0], size[1]))
            image = image.convert("RGB")
            image = np.asarray(image)

        LOG.debug('Image shape after loading %s', image.shape)

        self.current_image = image
        self.current_mag = mag
        self.current_downsample = self.mag / mag

        return image

    def tissue_mask_at(self, mag=5):
        """gets the tissue mask at a desired mag level
        """

        if self.current_mag == mag and self.current_image is not None:
            image = self.current_image
        else:
            image = self.image_at(mag)

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_inv = (255 - img_gray)  # invert the image intensity
        _, mask_ = cv2.threshold(
            img_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # contour, _ = cv2.findContours(mask_, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        result = cv2.findContours(
            mask_, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if len(result) == 2:
            contour = result[0]
        elif len(result) == 3:
            contour = result[1]

        for cnt in contour:
            cv2.drawContours(mask_, [cnt], 0, 255, -1)

        # --- removing small connected components ---
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
            mask_, connectivity=8)
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

        LOG.debug('tissue mask generated')

        return mask

    def annotation_mask_at(self, mag=5):
        """For generating annotated mask from the xml_annotations csv or xml file"""

        labels = np.linspace(0, len(self.annotation_labels) -
                             1, len(self.annotation_labels))

        if self.current_mag == mag and self.current_image is not None:
            image = self.current_image
        else:
            image = self.image_at(mag)

        image_shape = image.shape
        mask_annotated = np.zeros((image_shape[0], image_shape[1]), np.uint8)

        if self.annotation_file.endswith('.xml'):
            LOG.debug('xml file path : %s', self.annotation_file)
            dom = etree.parse(self.annotation_file)
            xml_annotations = dom.findall('xml_annotations')

            for annotation in xml_annotations:
                label = annotation.find('Regions/Region/Text').attrib['Value']
                LOG.debug('label : %s', label)
                if label not in self.annotation_labels:
                    # say if label is empty, then leave it or 3+2 kind
                    LOG.debug('%s was continued', label)
                    continue

                vertices = annotation.findall('Regions/Region/Vertices/Vertex')
                loc_temp = []
                for _, vertex in enumerate(vertices):
                    loc_x = int(vertex.attrib['X'])
                    loc_y = int(vertex.attrib['Y'])
                    loc_temp.append([loc_x, loc_y])

                ann_coordinates = loc_temp
                ann_coordinates = np.asarray(ann_coordinates)
                ann_coordinates = ann_coordinates / self.current_downsample
                ann_coordinates = ann_coordinates.astype(int)

                mask_annotated = cv2.drawContours(
                    mask_annotated,
                    [ann_coordinates],
                    0,
                    int(labels[np.where(label == self.annotation_labels)]), -1
                )  # filled with pixel corresponding to roi region

        elif self.annotation_file.endswith('.csv'):

            with open(self.annotation_file, 'r') as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    row = np.asarray(row)
                    label = str(row[0])
                    if label not in self.annotation_labels:  # say if label is empty, the leave it
                        LOG.debug('%s was continued', label)
                        continue
                    LOG.debug('label : %s', label)
                    ann_coordinates = row[1:]
                    ann_coordinates = ann_coordinates.astype(float)
                    ann_coordinates = ann_coordinates.astype(int)
                    ann_coordinates = np.reshape(
                        ann_coordinates, (int(len(ann_coordinates)/2), 2))
                    ann_coordinates = (
                        ann_coordinates/self.current_downsample).astype(int)

                    mask_annotated = cv2.drawContours(
                        mask_annotated, [ann_coordinates], 0,
                        int(labels[np.where(label == self.annotation_labels)]), -1
                    )
                    # filled with pixel corresponding to roi region

        return mask_annotated
