"""Annotation Readers for Histocartography."""
import logging
import sys
import csv
import cv2

import numpy as np

from lxml import etree

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::IO::Annotations')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
h1.setFormatter(formatter)
log.addHandler(h1)

default_labels = np.array(
    ['background', 'NROI', '3+3', '3+4', '4+3', '4+4', '4+5', '5+5'])


def get_annotation_mask(annotation_file,
                        image_shape,
                        scale_factor=1,
                        all_labels=default_labels):
    """For generating annotated mask from the annotation csv or xml file"""

    labels = np.linspace(0, len(all_labels) - 1, len(all_labels))
    labels_visualize = np.linspace(0, 255, len(all_labels))
    labels = labels.astype(int)
    labels_visualize = labels_visualize.astype(int)

    mask_annotated = np.zeros((image_shape[0], image_shape[1]), np.uint8)
    mask_annotated_visualize = np.zeros((image_shape[0], image_shape[1]),
                                        np.uint8)

    if (annotation_file.endswith('.xml')):
        log.debug('xml file path : {}'.format(annotation_file))
        dom = etree.parse(annotation_file)
        Annotation = dom.findall('Annotation')

        for j in range(len(Annotation)):
            label = Annotation[j].find('Regions/Region/Text').attrib['Value']
            log.debug('label : {}'.format(label))
            if label not in all_labels:
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
            ann_coordinates = ann_coordinates / scale_factor
            ann_coordinates = ann_coordinates.astype(int)

            mask_annotated = cv2.drawContours(
                mask_annotated, [ann_coordinates], 0,
                int(labels[np.where(label == all_labels)]),
                -1)  # filled with pixel corresponding to roi region
            mask_annotated_visualize = cv2.drawContours(
                mask_annotated_visualize, [ann_coordinates], 0,
                int(labels_visualize[np.where(label == all_labels)]), -1)

    elif (annotation_file.endswith('.csv')):

        with open(annotation_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                row = np.asarray(row)
                label = str(row[0])
                if label not in all_labels:
                    log.debug('{} was continued'.format(label))
                    continue
                log.debug('label : {}'.format(label))
                ann_coordinates = row[1:]
                ann_coordinates = ann_coordinates.astype(float)
                ann_coordinates = ann_coordinates.astype(int)
                ann_coordinates = np.reshape(
                    ann_coordinates, (int(len(ann_coordinates) / 2), 2))
                ann_coordinates = (ann_coordinates / scale_factor).astype(int)

                mask_annotated = cv2.drawContours(
                    mask_annotated, [ann_coordinates], 0,
                    int(labels[np.where(label == all_labels)]),
                    -1)  # filled with pixel corresponding to roi region
                mask_annotated_visualize = cv2.drawContours(
                    mask_annotated_visualize, [ann_coordinates], 0,
                    int(labels_visualize[np.where(label == all_labels)]),
                    -1)  # filled with pixel corresponding to roi region

    return mask_annotated, mask_annotated_visualize
