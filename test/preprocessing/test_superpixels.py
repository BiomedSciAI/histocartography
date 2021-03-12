"""Unit test for preprocessing.superpixel"""
import unittest
import numpy as np
import yaml
import os 
from PIL import Image
import shutil

from histocartography import PipelineRunner
from histocartography.preprocessing import SLICSuperpixelExtractor, ColorMergedSuperpixelExtractor


class SuperpixelTestCase(unittest.TestCase):
    """SuperpixelTestCase class."""

    @classmethod
    def setUpClass(self):
        self.data_path = os.path.join('..', 'data')
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '16B0001851_Block_Region_3.jpg'
        self.out_path = os.path.join(self.data_path, 'superpixel_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path) 
        os.makedirs(self.out_path)

    def test_slic_superpixel_extractor_with_pipeline_runner(self):
        """
        Test SLIC superpixel extractor with pipeline runner.
        """

        with open('config/slic_extractor.yml', 'r') as file:
            config = yaml.load(file)
        nr_superpixels = config['stages'][1]['superpixel']['params']['nr_superpixels']

        pipeline = PipelineRunner(output_path=self.out_path, save=False, **config)
        pipeline.precompute()
        output = pipeline.run(
            name=self.image_name.replace('.jpg', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        superpixels = output['superpixels']

        self.assertTrue(len(list(superpixels.shape))==2)              # mask is bi-dim
        self.assertTrue(isinstance(superpixels, np.ndarray))   # check type 
        self.assertTrue(np.unique(superpixels).shape[0] > nr_superpixels - 20) # check number of instances
        self.assertTrue(np.unique(superpixels).shape[0] < nr_superpixels + 20) # check number of instances 

        # 2. Re-run with existing output & ensure equal 
        output = pipeline.run(
            name=self.image_name.replace('.jpg', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        reload_superpixels = output['superpixels']

        self.assertTrue(np.array_equal(superpixels, reload_superpixels))

    def test_color_merged_superpixel_extractor_with_pipeline_runner(self):
        """
        Test color merged superpixel extractor with pipeline runner.
        """

        with open('config/color_merged_extractor.yml', 'r') as file:
            config = yaml.load(file)
        nr_superpixels = config['stages'][1]['superpixel']['params']['nr_superpixels']

        pipeline = PipelineRunner(output_path=self.out_path, save=True, **config)
        pipeline.precompute()
        output = pipeline.run(
            name=self.image_name.replace('.jpg', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        superpixels = output['superpixels']

        self.assertTrue(len(list(superpixels.shape))==2)  # mask is bi-dim
        self.assertTrue(isinstance(superpixels, np.ndarray))   # check type 
        self.assertTrue(np.unique(superpixels).shape[0] < nr_superpixels + 20) # check number of instances 

        # 2. Re-run with existing output & ensure equal 
        output = pipeline.run(
            name=self.image_name.replace('.jpg', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        reload_superpixels = output['superpixels']

        self.assertTrue(np.array_equal(superpixels, reload_superpixels))

    def test_slic_superpixel_extractor(self):
        """
        Test SLIC superpixel extractor. 
        """

        # 1. load the image
        image = np.array(Image.open(os.path.join(self.image_path, self.image_name)))

        # 2. run SLIC extraction
        nr_superpixels = 100
        slic_extractor = SLICSuperpixelExtractor(
            downsampling_factor=4,
            nr_superpixels=nr_superpixels
        )
        superpixels = slic_extractor.process(image)

        self.assertEqual(list(image.shape[:-1]), list(superpixels.shape))  # image HxW = mask HxW 
        self.assertTrue(len(list(superpixels.shape))==2)              # mask is bi-dim
        self.assertTrue(isinstance(superpixels, np.ndarray))
        self.assertTrue(np.unique(superpixels).shape[0] > nr_superpixels - 20)
        self.assertTrue(np.unique(superpixels).shape[0] < nr_superpixels + 20)

    def test_color_merged_superpixel_extractor(self):
        """
        Test Color Merged Superpixel extractor. 
        """

        # 1. load the image
        image = np.array(Image.open(os.path.join(self.image_path, self.image_name)))

        # 2. run color merger superpixel extraction 
        nr_superpixels = 100
        merging_extractor = ColorMergedSuperpixelExtractor(
            downsampling_factor=4,
            nr_superpixels=nr_superpixels
        )
        superpixels, _ = merging_extractor.process(image)

        self.assertEqual(list(image.shape[:-1]), list(superpixels.shape))  # image HxW = mask HxW 
        self.assertTrue(len(list(superpixels.shape))==2)              # mask is bi-dim
        self.assertTrue(isinstance(superpixels, np.ndarray))
        self.assertTrue(np.unique(superpixels).shape[0] < nr_superpixels + 20)


    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":

    unittest.main()
