"""Unit test for preprocessing.stain_normalizers"""
import unittest
import numpy as np
import cv2 
import torch 
import yaml
import dgl 
import os 
from PIL import Image
import shutil

from histocartography import PipelineRunner
from histocartography.preprocessing import MacenkoStainNormalizer, VahadaneStainNormalizer


class StainNormalizationTestCase(unittest.TestCase):
    """StainNormalizationTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '16B0001851_Block_Region_3.jpg'
        self.out_path = os.path.join(self.data_path, 'stain_normalization_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path) 
        os.makedirs(self.out_path)

    def test_macenko_normalizer(self):
        """
        Test Macenko Stain Normalization. 
        """

        image = np.array(Image.open(os.path.join(self.image_path, self.image_name)))

        stain_normalizer = MacenkoStainNormalizer(
            base_path=self.out_path,
            target=self.image_name.replace('.jpg', ''),
            target_path=os.path.join(self.image_path, self.image_name)
        )
        stain_normalizer.precompute(os.path.join(self.image_path, self.image_name))
        image_norm = stain_normalizer.process(image)

        self.assertTrue(isinstance(image_norm, np.ndarray))        # output is numpy 
        self.assertEqual(list(image.shape), list(image_norm.shape))  # image HxW = mask HxW 

    def test_vahadane_normalizer(self):
        """
        Test Vahadane Stain Normalization. 
        """

        image = np.array(Image.open(os.path.join(self.image_path, self.image_name)))

        stain_normalizer = VahadaneStainNormalizer(            
            base_path=self.out_path,
            target=self.image_name.replace('.jpg', ''),
            target_path=os.path.join(self.image_path, self.image_name)
        )
        stain_normalizer.precompute(os.path.join(self.image_path, self.image_name))
        image_norm = stain_normalizer.process(image)

        self.assertTrue(isinstance(image_norm, np.ndarray))        # output is numpy 
        self.assertEqual(list(image.shape), list(image_norm.shape))  # image HxW = mask HxW 

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    unittest.main()
