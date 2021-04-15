"""Unit test for preprocessing.tissue_mask"""
import unittest
import numpy as np
import yaml
import os
from PIL import Image
import shutil

from histocartography import PipelineRunner
from histocartography.preprocessing import GaussianTissueMask
from histocartography.utils import download_test_data


class TissueMaskTestCase(unittest.TestCase):
    """TissueMaskTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        download_test_data(self.data_path)
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '16B0001851_Block_Region_3.jpg'
        self.out_path = os.path.join(self.data_path, 'tissue_mask_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)

    def test_gaussian_tissue_mask_with_pipeline_runner(self):
        """
        Test gaussian tissue mask with pipeline runner.
        """

        # 1. Tissue mask detection with saving
        config_fname = os.path.join(
            self.current_path,
            'config',
            'tissue_mask',
            'tissue_mask.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)
        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.jpg', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        tissue_mask = output['tissue_mask']

        self.assertEqual(
            list(
                tissue_mask.shape), [
                1024, 1280])  # image HxW = mask HxW
        self.assertTrue(list(np.unique(tissue_mask))
                        == [0, 1])  # mask is binary
        # tissue pixel count large enough
        self.assertTrue(np.sum(tissue_mask) > 1e6)
        self.assertTrue(np.sum(tissue_mask) < 2e6)  # but not too large

        # 2. Re-run with existing output & ensure equal
        output = pipeline.run(
            output_name=self.image_name.replace('.jpg', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        reload_tissue_mask = output['tissue_mask']

        self.assertTrue(np.array_equal(tissue_mask, reload_tissue_mask))

    def test_gaussian_tissue_mask(self):
        """
        Test gaussian tissue mask.
        """

        # 1. load the image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))

        # 2. run tissue mask detection
        tissue_detecter = GaussianTissueMask(kernel_size=5)
        tissue_mask = tissue_detecter.process(image)

        # image HxW = mask HxW
        self.assertEqual(list(image.shape), [1024, 1280, 3])
        self.assertTrue(len(list(tissue_mask.shape))
                        == 2)      # mask is bi-dim
        self.assertEqual(image.shape[:-1],
                         tissue_mask.shape)  # image HxW = mask HxW
        self.assertTrue(list(np.unique(tissue_mask))
                        == [0, 1])  # mask is binary
        # tissue pixel count large enough
        self.assertTrue(np.sum(tissue_mask) > 1e6)
        self.assertTrue(np.sum(tissue_mask) < 2e6)  # but not too large

        # 3. save tissue mask
        tissue_mask = Image.fromarray(np.uint8(tissue_mask * 255))
        tissue_mask.save(
            os.path.join(
                self.out_path,
                self.image_name.replace(
                    '.jpg',
                    '.png')),
        )

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":

    unittest.main()
