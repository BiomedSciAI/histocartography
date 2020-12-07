"""Unit test for preprocessing.nuclei_extraction"""
import unittest
import numpy as np
import cv2 

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
        image = np.array(load_image('../data/1937_benign_4.png'))

        # 2. create a nuclei extractor 
        extractor = NucleiExtractor(
            model_path='s3://mlflow/6f8ad1831d1846d8bb055ed5ffb24056/artifacts/hovernet_pannuke'
        )

        # 3. process the image 
        instance_map, instance_labels = extractor.process(image)

        # 4. viz the output
        overlaid_output = visualize_instances(image, instance_map, instance_labels)
        overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
        cv2.imwrite('1937_benign_4_nuclei_prediction.png', overlaid_output)

    def test_nuclei_extractor_with_local(self):
        """Test nuclei extraction with local model."""

        image = np.array(load_image('../data/1937_benign_4.png'))
        extractor = NucleiExtractor(
            model_path='checkpoints/hovernet_pannuke.pth'
        )
        instance_map, instance_labels = extractor.process(image)

        overlaid_output = visualize_instances(image, instance_map, instance_labels)
        overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
        cv2.imwrite('1937_benign_4_nuclei_prediction.png', overlaid_output)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    model = NucleiExtractionTestCase()
    model.test_nuclei_extractor_with_local()
    model.test_nuclei_extractor_with_mlflow()
