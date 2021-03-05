"""Unit test for preprocessing.nuclei_concept_extraction"""
import unittest
import numpy as np
import cv2
import h5py
import os 

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
        base_path = '../data'
        image_name = '283_dcis_4.png'
        image = np.array(load_image(os.path.join(base_path, 'images', image_name)))

        # 2. detect nuclei
        extractor = NucleiExtractor(
            model_path='checkpoints/hovernet_kumar_notype.pth'
        )
        instance_map, _ = extractor.process(image)

        # 3. extract nuclei concepts
        nuclei_concept_extractor = NucleiConceptExtractor(
            concept_names='area,perimeter,roughness,eccentricity,roundness,shape_factor,mean_crowdedness,std_crowdedness,glcm_dissimilarity,glcm_contrast,glcm_homogeneity,glcm_ASM,glcm_entropy,glcm_dispersion'
        )
        concepts = nuclei_concept_extractor.process(image, instance_map)

        print('Nuclei concepts:', concepts.shape)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    model = NucleiConceptExtractionTestCase()
    model.test_concept_extractor()
