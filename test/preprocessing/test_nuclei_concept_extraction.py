"""Unit test for preprocessing.nuclei_concept_extraction"""
import unittest
import numpy as np
import cv2 
import h5py 

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
        image = np.array(load_image('../data/283_dcis_4.png'))

        # 2. detect nuclei
        extractor = NucleiExtractor(
            model_path='checkpoints/hovernet_pannuke.pth'
        )
        instance_map, _, _ = extractor.process(image)

        # 3. extract nuclei concepts 
        nuclei_concept_extractor = NucleiConceptExtractor(
            concept_names='area,perimeter,roundness,ellipticity,eccentricity'
        )
        concepts = nuclei_concept_extractor.process(image, instance_map)

        print('Nuclei concepts:', concepts.shape)

        # 4. save as h5 file 
        with h5py.File('283_dcis_4_concepts.h5', 'w') as hf:
            hf.create_dataset("concepts",  data=concepts)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    model = NucleiConceptExtractionTestCase()
    model.test_concept_extractor()
