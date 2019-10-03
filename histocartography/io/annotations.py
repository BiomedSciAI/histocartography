"""Annotation Readers for Histocartography."""
import logging
import sys
import cv2
import csv
import numpy as np
from scipy.sparse import coo_matrix

from lxml import etree

from abc import ABC, abstractmethod

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::IO::Annotations')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
h1.setFormatter(formatter)
log.addHandler(h1)

DEFAULT_LABELS = [
    'background', 'NROI', '3+3', '3+4', '4+3', '4+4', '4+5', '5+5'
]



class Annotation(ABC):

    def __init__(self):
        self.annotation_labels = [None]

    @abstractmethod
    def mask(self, size, origin):
        pass


class XMLAnnotation(Annotation):

    def __init__(self, annotation_file, annotation_labels=DEFAULT_LABELS):
        """
            annotation_file (path): path to XML annotation file
            annotation_labels (list): list of label names
        """
        super().__init__()
        self.annotation_file = annotation_file
        self.annotation_labels = annotation_labels
        log.debug('xml file path : %s', self.annotation_file)
        dom = etree.parse(self.annotation_file)
        self.xml_annotations = dom.findall('Annotation')

        annotated_region_generator = (
            self._get_region_pixels_from_xml(annotation)
            for annotation in self.xml_annotations
        )
        self.pixels = np.vstack(
            [
                annotated_region
                for annotated_region in annotated_region_generator
                if annotated_region is not None
            ]
        )

    def _get_region_pixels_from_xml(self, annotation):
        """
            annotation (etree): xml tree containin a single annotation region
        """
        region_pixels = None
        label_name = annotation.find('Regions/Region/Text').attrib['Value']
        log.debug('label_name : %s', label_name)
        if label_name in self.annotation_labels:
            label_index = self.annotation_labels.index(label_name)

            vertices = annotation.findall('Regions/Region/Vertices/Vertex')
            v_coords = np.ndarray((len(vertices), 2), dtype=int)

            for vertex_i, vertex in enumerate(vertices):
                v_coords[vertex_i, 0] = int(vertex.attrib['X'])
                v_coords[vertex_i, 1] = int(vertex.attrib['Y'])
            origin_shift = np.amin(v_coords, axis=0)
            region_size = np.amax(v_coords, axis=0) - origin_shift
            normalized_coords = v_coords - origin_shift

            log.debug('Region Size %s', region_size)

            if len(normalized_coords) > 2:
                region = np.zeros((region_size[0], region_size[1]), np.uint8)
                region = cv2.drawContours(
                    region, [normalized_coords], -1, label_index, -1
                )
                sparse_region = coo_matrix(region)

                region_pixels = np.array(
                    [
                        [x, y, value] for x, y, value in zip(
                            sparse_region.row +
                            origin_shift[0], sparse_region.col +
                            origin_shift[1], sparse_region.data
                        )
                    ]
                )

        else:
            # skips regions with no valid label
            log.debug(f'{label_name} was not found in annotation label list')

        return region_pixels

    def mask(self, size, origin=(0, 0), selected_downsample=1):
        """
            Generates mask for image of a given shape
            size (tuple) : expected size of the annotation
            origin (tuple) : origin of the mask patch in the ORIGINAL SPACE
            selected_downsample (float): scale of downsampling required
        
        """

        x = origin[0]
        y = origin[1]

        annotation_mask = np.zeros((size[0], size[1]), np.uint8)

        valid_annotations_x = np.where(
            np.isin(
                self.pixels[:, 0],
                range(x, x + int(size[0] * selected_downsample)),
                assume_unique=False
            )
        )

        valid_annotations_y = np.where(
            np.isin(
                self.pixels[:, 1],
                range(y, y + int(size[1] * selected_downsample)),
                assume_unique=False
            )
        )

        valid_annotations = self.pixels[
            np.intersect1d(valid_annotations_x, valid_annotations_y), :]

        valid_annotations[:, 0] = (
            valid_annotations[:, 0] - x
        ) / selected_downsample
        valid_annotations[:, 1] = (
            valid_annotations[:, 1] - y
        ) / selected_downsample

        valid_annotations = valid_annotations.astype(int)

        annotation_mask[valid_annotations[:, 1], valid_annotations[:, 0]
                        ] = valid_annotations[:, 2]

        return annotation_mask


class CSVAnnotation(XMLAnnotation):

    def __init__(self, annotation_file, annotation_labels=DEFAULT_LABELS):
        """
            annotation_file (path): path to CSV annotation file
            annotation_labels (list): list of label names
        """
        super().__init__()
        self.annotation_file = annotation_file
        self.annotation_labels = annotation_labels

        with open(self.annotation_file, 'r') as csv_file:
            reader = csv.reader(csv_file)
            annotated_region_generator = (
                self._get_region_pixels_from_csv_row(row) for row in reader
            )

            self.pixels = np.vstack(
                [
                    annotated_region
                    for annotated_region in annotated_region_generator
                    if annotated_region is not None
                ]
            )

    def _get_region_pixels_from_csv_row(self, row):
        """
            gets all annotated pixels
        """
        region_pixels = None

        row = np.asarray(row)
        label = str(row[0])
        if label not in self.annotation_labels:
            log.debug(f'{label} was not found in the labels list')
        else:
            label_index = self.annotation_labels.index(label)

            log.debug(f'Label: {label}')
            ann_coordinates = row[1:]
            ann_coordinates = ann_coordinates.astype(float)
            ann_coordinates = ann_coordinates.astype(int)
            ann_coordinates = np.reshape(
                ann_coordinates, (int(len(ann_coordinates) / 2), 2)
            )

            pixel_count, _ = ann_coordinates.shape
            region_pixels = label_index * np.ones((pixel_count, 3))
            region_pixels[:, 0:2] = ann_coordinates

        return region_pixels


class ImageAnnotation(Annotation):

    def __init__(self, annotation_file, annotation_labels=DEFAULT_LABELS):
        super().__init__()
        self.annotation_file = annotation_file
        self.annotation_labels = annotation_labels
        log.debug(f'Annotation file: {annotation_file}')
        mask_annotated = cv2.imread(
            annotation_file, 0
        )  # read image in gray scale
        self.mask_annotated = np.array(mask_annotated, dtype=np.uint8)

    def mask(self, size, origin=(0, 0), selected_downsample=1):

        original_size = (
            int(size[0] * selected_downsample),
            int(size[1] * selected_downsample)
        )

        region = self.mask_annotated[origin[0]:origin[0] +
                                     original_size[0], origin[1]:origin[1] +
                                     original_size[1]]

        try:
            mask = cv2.resize(region, (size[0], size[1]))
        except:
            log.warning('Could not resize patch')
            log.debug(f'FILE: {self.annotation_file}')
            log.debug(f'FULL SIZE: {self.mask_annotated.shape}')
            log.debug(f'REGION SHAPE: {region.shape}')
            log.debug(f'SIZE: {size}')
            log.debug(f'ORIGIN: {origin}')
            log.debug(f'ORIGINAL SIZE: {original_size}')
            log.debug(f'SELECTED_DOWNSAMPLE: {selected_downsample}')
            mask = np.zeros(size)

        return mask
