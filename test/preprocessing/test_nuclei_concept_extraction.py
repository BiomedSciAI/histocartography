"""Unit test for preprocessing.nuclei_concept_extraction"""
import unittest
import numpy as np
import cv2
import h5py
import os
import shutil
from PIL import Image

from histocartography import PipelineRunner
from histocartography.preprocessing import NucleiExtractor, H5Loader
from histocartography.preprocessing import NucleiConceptExtractor
from histocartography.utils import download_test_data

import time 

class NucleiConceptExtractionTestCase(unittest.TestCase):
    """NucleiConceptExtractionTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        download_test_data(self.data_path)
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '283_dcis_4.png'
        self.nuclei_map_path = os.path.join(self.data_path, 'nuclei_maps')
        self.nuclei_map_name = '283_dcis_4.h5'
        self.out_path = os.path.join(
            self.data_path, 'nuclei_concept_extraction_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)

    def test_concept_extractor(self):
        """Test nuclei concept extraction."""

        # 1. load an image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))

        # 2. load nuclei
        h5_loader = H5Loader()
        instance_map, instance_centroids = h5_loader._process(
            path=os.path.join(
                self.nuclei_map_path,
                self.nuclei_map_name
            )
        )

        # 3. extract nuclei concepts
        nuclei_concept_extractor = NucleiConceptExtractor()
        concepts = nuclei_concept_extractor.process(image, instance_map)

        self.assertIsInstance(concepts, np.ndarray)  # check type is np array
        self.assertEqual(concepts.shape[0], 331)  # check number of instances
        # check number of node features
        self.assertEqual(concepts.shape[1], 24)

    def test_concept_extractor_with_list(self):
        """Test nuclei extraction with list of concepts."""

        # 1. load an image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))

        # 2. load nuclei
        h5_loader = H5Loader()
        instance_map, instance_centroids = h5_loader._process(
            path=os.path.join(
                self.nuclei_map_path,
                self.nuclei_map_name
            )
        )

        # 3. extract nuclei concepts
        nuclei_concept_extractor = NucleiConceptExtractor(concept_names='area,eccentricity')
        concepts = nuclei_concept_extractor.process(image, instance_map)

        self.assertIsInstance(concepts, np.ndarray)  # check type is np array
        self.assertEqual(concepts.shape[0], 331)  # check number of instances
        # check number of node features
        self.assertEqual(concepts.shape[1], 2)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    unittest.main()
