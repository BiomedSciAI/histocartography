"""Unit test for preprocessing.nuclei_concept_extraction"""
import unittest
import numpy as np
import cv2 

from histocartography.preprocessing.nuclei_extraction import NucleiExtractor
from histocartography.preprocessing.nuclei_concept_extraction import NucleiConceptExtractor
from histocartography.utils.io import load_image
from histocartography.utils.hover import visualize_instances


class NucleiConceptExtractionTestCase(unittest.TestCase):
    """NucleiConceptExtractionTestCase class."""

    def setUp(self):
        """Setting up the test."""

    def test_concept_extractor(self):
        """Test nuclei extraction with local model."""

        # 1. load image
        image = np.array(load_image('../data/1937_benign_4.png'))

        # @TODO: debug purposes 
        nuclei_concept_extractor = NucleiConceptExtractor(concept_names='area,perimeter')

        # 2. detect nuclei
        extractor = NucleiExtractor(
            model_path='checkpoints/hovernet_pannuke.pth'
        )
        instance_map, instance_labels, instance_centroids = extractor.process(image)

        # 3. extract nuclei concepts 
        concepts = nuclei_concept_extractor.process(image, instance_map)

        print('Nuclei concepts:', concepts.shape)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    model = NucleiConceptExtractionTestCase()
    model.test_concept_extractor()
