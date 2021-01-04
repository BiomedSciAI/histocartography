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
        image = np.array(load_image('../data/283_dcis_4.png'))

        # 2. create a nuclei extractor 
        extractor = NucleiExtractor(
            model_path='s3://mlflow/7653a1e1e5f443dd81ff61aa59386bd0/artifacts/hovernet_kumar_notype'
        )

        # 3. process the image 
        instance_map, instance_centroids = extractor.process(image)

        # 4. viz the output
        overlaid_output = visualize_instances(image, instance_map)
        overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
        cv2.imwrite('283_dcis_4_nuclei_prediction.png', overlaid_output)

    def test_nuclei_extractor_with_local(self):
        """Test nuclei extraction with local model."""

        image = np.array(load_image('../data/283_dcis_4.png'))
        extractor = NucleiExtractor(
            model_path='checkpoints/hovernet_kumar_notype.pth'
        )
        instance_map, instance_centroids = extractor.process(image)

        overlaid_output = visualize_instances(image, instance_map)
        overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
        cv2.imwrite('283_dcis_4_nuclei_prediction.png', overlaid_output)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    model = NucleiExtractionTestCase()
    model.test_nuclei_extractor_with_local()
    # model.test_nuclei_extractor_with_mlflow()
