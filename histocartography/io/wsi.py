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
import cv2
from PIL import Image


#from PIL import Image, ImageDraw

# setup logging
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::IO::WSI')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
h1.setFormatter(formatter)
log.addHandler(h1)


safe_vendors = ['aperio','hamamatsu','leica','mirax','sakura','ventana']
# mapping of the magnification property used by the vendor
# usage:
# if self.stack.properties['openslide.vendor'] in self.safe_vendors :
    # magnification = self.stack.properties[self.magnification_tag[self.stack.properties['openslide.vendor']]]
    # magnification = self.stack.properties['openslide.objective-power']
# self.stack.properties[magn_tag[self.stack.properties['openslide.vendor']]]
# 
magnification_tag = {
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

default_labels = np.array(['background', 'NROI', '3+3', '3+4', '4+3', '4+4', '4+5', '5+5'])


class WSI:

    def __init__(self, wsi_file, annotation_file=None, annotation_labels = default_labels):
        """Constructs a WSI object with a given wsi_file

        Parameters
        ----------
        wsi_file : str
            The file containing the slide
        annotation_file : str, optional
            Annotations for the file
        """

        self.wsi_file = wsi_file
        self.annotation_file = annotation_file
        self.annotation_labels = annotation_labels
        self.current_image = None
        self.current_magnification = None

        log.debug('wsi_file : {}'.format(self.wsi_file))


        if os.path.isfile(self.wsi_file):
            self.stack = open_slide(self.wsi_file)
            properties = self.stack.properties
            self.vendor = properties['openslide.vendor']
        else:
            log.error('File does not exist')

        if self.stack.properties['openslide.vendor'] in safe_vendors:
            self.vendor_magnification = properties[magnification_tag[self.vendor]]
            self.openslide_magnification = properties['openslide.objective-power']
            self.magnification = float(self.openslide_magnification)
        else:
            self.magnification = None

        
        self.downsamples = np.rint(self.stack.level_downsamples).astype(int)
        self.available_magnifications = [ self.magnification/np.rint(downsample) for downsample in self.downsamples]
      

        log.debug('Original magnification: {}'.format(self.vendor_magnification))
        log.debug('Original magnification (openslide): {}'.format(self.openslide_magnification))
        log.debug('Levels: {}'.format(self.stack.level_count))
        log.debug('Level dimensions {}'.format(self.stack.level_dimensions))
        log.debug('Downsamples: {}'.format(self.downsamples))
        log.debug('Possible resolutions: {}'.format(self.available_magnifications))
        
        

    def image_at(self, magnification=5):
        """gets the image at a desired magnification level
        """

        log.debug('Downsample for desired resolution : {}'.format(self.magnification / magnification))
        level = self.stack.get_best_level_for_downsample(self.magnification / magnification)

        '''
        # include condition for when the desired resolution doesnot exist
        try :
            level = np.where(possible_resolutions == magnification)[0][0]

        except IndexError:
            log.debug('Level for desired resolution doesnot exist in the self.stack : {}'.format(magnification))
            return # fix this
        '''

        log.debug('Level for desired resolution : {}'.format(level))

        size = self.stack.level_dimensions[0]
        size = self.stack.level_dimensions[level]


        if (level <= 2):
            x = size[0]
            y = size[1]
            log.debug('Expected size of image: {},{}'.format(y, x))
            image = np.empty([y, x, 3], dtype=np.uint8)
            x_13_0 = int(size_0[0] / 3)
            y_13_0 = int(size_0[1] / 3)
            x_23_0 = 2 * x_13_0
            y_23_0 = 2 * y_13_0
            x_13 = int(size[0] / 3)
            y_13 = int(size[1] / 3)
            x_23 = 2 * x_13
            y_23 = 2 * y_13
            img_1 = np.asarray((self.stack.read_region((0, 0), level, (x_13, y_13))).convert("RGB"))
            img_2 = np.asarray((self.stack.read_region((x_13_0, 0), level, (x_13, y_13))).convert("RGB"))
            img_3 = np.asarray((self.stack.read_region((x_23_0, 0), level, ((x - x_23), y_13))).convert("RGB"))

            img_4 = np.asarray((self.stack.read_region((0, y_13_0), level, (x_13, y_13))).convert("RGB"))
            img_5 = np.asarray((self.stack.read_region((x_13_0, y_13_0), level, (x_13, y_13))).convert("RGB"))
            img_6 = np.asarray((self.stack.read_region((x_23_0, y_13_0), level, ((x - x_23), y_13))).convert("RGB"))

            img_7 = np.asarray((self.stack.read_region((0, y_23_0), level, (x_13, (y - y_23)))).convert("RGB"))
            img_8 = np.asarray((self.stack.read_region((x_13_0, y_23_0), level, (x_13, (y - y_23)))).convert("RGB"))
            img_9 = np.asarray((self.stack.read_region((x_23_0, y_23_0), level, ((x - x_23), (y - y_23)))).convert("RGB"))

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
            image = self.stack.read_region((0, 0), level, (size[0], size[1]))
            image = image.convert("RGB")
            image = np.asarray(image)

        log.debug('Image shape after loading : {}'.format(image.shape))

        self.current_image = image
        self.current_magnification = magnification
        self.current_downsample = self.magnification / magnification

        return image

    def tissue_mask_at(self, magnification=5):

        if self.current_magnification == magnification and self.current_image is not None:
            image = self.current_image
        else:
            image = self.image_at(magnification)
        
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_inv = (255 - img_gray)  # invert the image intensity
        val_thr_stained, mask_ = cv2.threshold(img_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #contour, _ = cv2.findContours(mask_, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        result = cv2.findContours(mask_, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if len(result)==2:
            contour = result[0]
        elif len(result)==3:
            contour = result[1]


        for cnt in contour:
            cv2.drawContours(mask_, [cnt], 0, 255, -1)

        # --- removing small connected components ---
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        mask_remove_small = np.zeros((output.shape))
        remove_blob_size = 5000 #

        for i in range(0, nb_components):
            if sizes[i] >= remove_blob_size:
                mask_remove_small[output == i + 1] = 255

        mask_remove_small = mask_remove_small.astype(int)
        mask_remove_small = np.uint8(mask_remove_small)

        mask = np.zeros((mask_.shape[0], mask_.shape[1]), np.uint8)
        mask[mask_remove_small == 255] = 255  # NROI

        log.debug('tissue mask generated')


        return mask


    def annotation_mask_at(self, magnification=5):
        """For generating annotated mask from the annotation csv or xml file"""
        
        labels = np.linspace(0, len(self.annotation_labels) - 1, len(self.annotation_labels))
        
        if self.current_magnification == magnification and self.current_image is not None:
            image = self.current_image
        else:
            image = self.image_at(magnification)

        image_shape = image.shape
        mask_annotated = np.zeros((image_shape[0], image_shape[1]), np.uint8)

        if(self.annotation_file.endswith('.xml')):
            log.debug('xml file path : {}'.format(self.annotation_file))
            dom = etree.parse(self.annotation_file)
            Annotation = dom.findall('Annotation')

            for j in range(len(Annotation)):
                label = Annotation[j].find('Regions/Region/Text').attrib['Value']
                log.debug('label : {}'.format(label))
                if (label not in self.annotation_labels): # say if label is empty, then leave it or 3+2 kind
                    log.debug('{} was continued'.format(label))
                    continue

                vertices = Annotation[j].findall('Regions/Region/Vertices/Vertex')
                loc_temp = []
                for counter, x in enumerate(vertices):
                    loc_X = int(x.attrib['X'])
                    loc_Y = int(x.attrib['Y'])
                    loc_temp.append([loc_X, loc_Y])

                ann_coordinates = loc_temp
                ann_coordinates = np.asarray(ann_coordinates)
                ann_coordinates = ann_coordinates / self.current_downsample
                ann_coordinates = ann_coordinates.astype(int)

                mask_annotated = cv2.drawContours(mask_annotated, [ann_coordinates], 0, int(labels[np.where(label == self.annotation_labels)]), -1)  # filled with pixel corresponding to roi region


        elif(self.annotation_file.endswith('.csv')):

            with open(self.annotation_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    row = np.asarray(row)
                    label = str(row[0])
                    if (label not in self.annotation_labels):  # say if label is empty, the leave it
                        log.debug('{} was continued'.format(label))
                        continue
                    log.debug('label : {}'.format(label))
                    ann_coordinates = row[1:]
                    ann_coordinates = ann_coordinates.astype(float)
                    ann_coordinates = ann_coordinates.astype(int)
                    ann_coordinates = np.reshape(ann_coordinates, (int(len(ann_coordinates)/2) , 2))
                    ann_coordinates = (ann_coordinates/self.current_downsample).astype(int)

                    mask_annotated = cv2.drawContours(mask_annotated, [ann_coordinates], 0, int(labels[np.where(label == self.annotation_labels)]), -1) # filled with pixel corresponding to roi region

        return mask_annotated
