"""Unit test for preprocessing.nuclei_extraction"""
import unittest
import numpy as np
import cv2 
import os 
import shutil

from histocartography.preprocessing import NucleiExtractor
from histocartography.visualisation import GraphVisualization
from histocartography.utils.io import load_image


class NucleiExtractionTestCase(unittest.TestCase):
    """NucleiExtractionTestCase class."""

    @classmethod
    def setUpClass(self):
        self.data_path = os.path.join('..', 'data')
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '283_dcis_4.png'
        self.out_path = os.path.join(self.data_path, 'nuclei_extraction_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path) 
        os.makedirs(self.out_path)

    def test_nuclei_extractor_with_local_model(self):
        """Test nuclei extraction with local model."""

        # 1. load an image
        image_name = '283_dcis_4.png'
        image = np.array(load_image(os.path.join(self.image_path, self.image_name)))

        # 2. create a nuclei extractor 
        extractor = NucleiExtractor(
            model_path='../../histocartography/ml/models/hovernet_monusac.pt'
        )

        # 3. process the image 
        instance_map, instance_centroids = extractor.process(image)

        print('Instance map:', np.unique(instance_map))
        print('Instance centroids:', np.unique(instance_centroids))

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    unittest.main()