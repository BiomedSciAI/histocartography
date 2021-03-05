"""Unit test for preprocessing.nuclei_extraction"""
import unittest
import numpy as np
import cv2 
import os 

from histocartography.preprocessing.nuclei_extraction import NucleiExtractor
from histocartography.utils.io import load_image
from histocartography.utils.hover import visualize_instances


class NucleiExtractionTestCase(unittest.TestCase):
    """NucleiExtractionTestCase class."""

    def setUp(self):
        """Setting up the test."""

    def test_nuclei_extractor_with_mlflow(self):
        """Test nuclei extraction with MLflow model."""

        # 1. load an image
        base_path = '../data'
        image_name = '283_dcis_4.png'
        image = np.array(load_image(os.path.join(base_path, 'images', image_name)))

        # 2. create a nuclei extractor 
        extractor = NucleiExtractor(
            model_path='s3://mlflow/90a7e42bf0224683933bdc4bcb496a24/artifacts/hovernet_kumar_notype'
        )

        # 3. process the image 
        instance_map, instance_centroids = extractor.process(image)

    def test_nuclei_extractor_with_local(self):
        """Test nuclei extraction with local model."""

        image = np.array(load_image('../data/283_dcis_4.png'))
        extractor = NucleiExtractor(
            model_path='checkpoints/hovernet_kumar_notype.pth'
        )
        instance_map, instance_centroids = extractor.process(image)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    model = NucleiExtractionTestCase()
    model.test_nuclei_extractor_with_local()
    model.test_nuclei_extractor_with_mlflow()
