"""Unit test for preprocessing.superpixel"""
import unittest
import numpy as np
import yaml
import os
import shutil
import pandas as pd

from histocartography import PipelineRunner, BatchPipelineRunner
from histocartography.utils import download_test_data


class SuperpixelTestCase(unittest.TestCase):
    """SuperpixelTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        download_test_data(self.data_path)
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '283_dcis_4.png'
        self.out_path = os.path.join(self.data_path, 'superpixel_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)

    def test_slic_superpixel_extractor_with_pipeline_runner(self):
        """
        Test SLIC superpixel extractor with pipeline runner.
        """

        config_fname = os.path.join(
            self.current_path,
            'config',
            'superpixels',
            'slic_extractor.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=None, **config)
        output = pipeline.run(
            output_name=None,
            image_path=os.path.join(self.image_path, self.image_name)
        )
        superpixels = output['superpixels']

        self.assertTrue(
            isinstance(
                superpixels,
                np.ndarray))        # check type
        self.assertEqual(len(list(superpixels.shape)),
                         2)           # mask is bi-dim
        # check number of instances
        self.assertEqual(len(np.unique(superpixels)), 81)

        # Re-run with existing output & ensure equal
        output = pipeline.run(
            output_name=None,
            image_path=os.path.join(self.image_path, self.image_name)
        )
        reload_superpixels = output['superpixels']

        self.assertTrue(np.array_equal(superpixels, reload_superpixels))

    def test_color_merged_superpixel_extractor_with_pipeline_runner(self):
        """
        Test color merged superpixel extractor with pipeline runner.
        """

        config_fname = os.path.join(
            self.current_path,
            'config',
            'superpixels',
            'color_merged_extractor.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        superpixels = output['merged_superpixels_map']

        self.assertTrue(
            isinstance(
                superpixels,
                np.ndarray))        # check type
        self.assertEqual(len(list(superpixels.shape)),
                         2)           # mask is bi-dim
        # check number of instances
        self.assertEqual(len(np.unique(superpixels)), 23)

        # Re-run with existing output & ensure equal
        output = pipeline.run(
            output_name=self.image_name.replace('.jpg', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        reload_superpixels = output['merged_superpixels_map']

        self.assertTrue(np.array_equal(superpixels, reload_superpixels))

    def test_slic_superpixel_extractor_with_batch_pipeline_runner(self):
        """
        Test SLIC superpixel extractor with batch pipeline runner.
        """

        config_fname = os.path.join(
            self.current_path,
            'config',
            'superpixels',
            'slic_extractor.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        metadata = pd.DataFrame(
            {'image_path': [os.path.join(self.image_path, self.image_name)]})
        pipeline = BatchPipelineRunner(
            save_path=self.out_path,
            pipeline_config=config)
        output = pipeline.run(metadata=metadata, return_out=True)
        superpixels = output[0]['superpixels']

        self.assertTrue(isinstance(superpixels, np.ndarray))    # check type
        self.assertEqual(len(list(superpixels.shape)),
                         2)       # mask is bi-dim
        # check number of instances
        self.assertEqual(len(np.unique(superpixels)), 81)

        pipeline.run(metadata=metadata, cores=2)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":

    unittest.main()
