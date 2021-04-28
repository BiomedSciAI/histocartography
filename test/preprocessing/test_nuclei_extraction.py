"""Unit test for preprocessing.nuclei_extraction"""
import unittest
import numpy as np
import cv2
import os
import shutil
from PIL import Image
import matplotlib
import yaml

from histocartography import PipelineRunner
from histocartography.preprocessing import NucleiExtractor
from histocartography.utils import download_test_data


class NucleiExtractionTestCase(unittest.TestCase):
    """NucleiExtractionTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        download_test_data(self.data_path)
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '283_dcis_4.png'
        self.out_path = os.path.join(self.data_path, 'nuclei_extraction_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)

    def test_nuclei_extractor_with_pipeline_runner(self):
        """Test nuclei extraction with local model."""

        config_fname = os.path.join(
            self.current_path,
            'config',
            'nuclei_extraction',
            'nuclei_extractor.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        instance_map = output['nuclei_map']
        instance_centroids = output['nuclei_centroids']

        # 3. run tests
        self.assertTrue(isinstance(instance_map, np.ndarray))
        self.assertTrue(isinstance(instance_centroids, np.ndarray))
        self.assertEqual(len(instance_centroids), 331)

    def test_nuclei_extractor_with_monusac(self):
        """Test nuclei extraction with monusac model."""

        # 1. load an image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))

        # 2. extract nuclei
        extractor = NucleiExtractor(
            pretrained_data='monusac'
        )
        instance_map, instance_centroids = extractor.process(image)

        # 3. run tests
        self.assertEqual(instance_map.shape[0], image.shape[0])
        self.assertEqual(instance_map.shape[1], image.shape[1])
        self.assertEqual(len(instance_centroids), 134)

    def test_nuclei_extractor_with_pannuke(self):
        """Test nuclei extraction with pannuke model."""

        # 1. load an image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))

        # 2. extract nuclei
        extractor = NucleiExtractor(
            pretrained_data='pannuke'
        )
        instance_map, instance_centroids = extractor.process(image)

        # 3. run tests
        self.assertEqual(instance_map.shape[0], image.shape[0])
        self.assertEqual(instance_map.shape[1], image.shape[1])
        self.assertEqual(len(instance_centroids), 331)

    def test_nuclei_extractor_with_bs(self):
        """Test nuclei extraction with specified batch size."""

        # 1. load an image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))

        # 2. extract nuclei
        extractor = NucleiExtractor(
            batch_size=4
        )
        instance_map, instance_centroids = extractor.process(image)

        # 3. run tests
        self.assertEqual(instance_map.shape[0], image.shape[0])
        self.assertEqual(instance_map.shape[1], image.shape[1])
        self.assertEqual(len(instance_centroids), 331)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    unittest.main()
