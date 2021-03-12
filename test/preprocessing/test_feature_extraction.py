"""Unit test for preprocessing.feature_extraction"""
import unittest
import numpy as np
import yaml
import os 
import torch 
from PIL import Image
import shutil

from histocartography import PipelineRunner
from histocartography.preprocessing import HandcraftedFeatureExtractor, DeepTissueFeatureExtractor
from histocartography.preprocessing import AugmentedDeepTissueFeatureExtractor, FeatureMerger
from histocartography.preprocessing import ColorMergedSuperpixelExtractor


class FeatureExtractionTestCase(unittest.TestCase):
    """FeatureExtractionTestCase class."""

    @classmethod
    def setUpClass(self):
        self.data_path = os.path.join('..', 'data')
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '283_dcis_4.png'
        self.out_path = os.path.join(self.data_path, 'feature_extraction_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path) 
        os.makedirs(self.out_path)

    def test_handcrafted_feature_extractor_with_pipeline_runner(self):
        """
        Test handcrafted feature extractor with pipeline runner.
        """

        with open('config/handcrafted_feature_extractor.yml', 'r') as file:
            config = yaml.load(file)

        pipeline = PipelineRunner(output_path=self.out_path, save=True, **config)
        pipeline.precompute()
        output = pipeline.run(
            name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        features = output['features']

        self.assertTrue(isinstance(features, torch.Tensor))  # check type 
        self.assertEqual(features.shape[1], 65)  # check number of features per instance  
        self.assertEqual(features.shape[0], 9)  # check number of instances detected 

        # 2. Re-run with existing output & ensure equal 
        output = pipeline.run(
            name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        reload_features = output['features']

        self.assertTrue(np.array_equal(features, reload_features))

    def test_handcrafted_feature_extractor(self):
        """
        Test Handcrafted feature extractor. 
        """

        # 1. load the image
        image = np.array(Image.open(os.path.join(self.image_path, self.image_name)))

        # 2. run SLIC extraction
        nr_superpixels = 100
        slic_extractor = ColorMergedSuperpixelExtractor(
            downsampling_factor=4,
            nr_superpixels=nr_superpixels
        )
        superpixels, _ = slic_extractor.process(image)

        # 3. run handcrafted feature extraction
        feature_extractor = HandcraftedFeatureExtractor()
        features = feature_extractor.process(image, superpixels)

        self.assertTrue(isinstance(features, torch.Tensor))  # check type 
        self.assertEqual(features.shape[1], 65)  # check number of features per instance  
        self.assertEqual(features.shape[0], 9)  # check number of instances detected 

    def test_deep_tissue_extractor_with_pipeline_runner(self):
        """
        Test deep tissue feature extractor with pipeline runner.
        """

        with open('config/deep_tissue_feature_extractor.yml', 'r') as file:
            config = yaml.load(file)

        pipeline = PipelineRunner(output_path=self.out_path, save=True, **config)
        pipeline.precompute()
        output = pipeline.run(
            name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        features = output['features']

        self.assertTrue(isinstance(features, torch.Tensor))  # check type 
        self.assertEqual(features.shape[0], 1)  # check number of augmentations (only 1 expected)
        self.assertEqual(features.shape[1], 10)  # check number horizontal patches 
        self.assertEqual(features.shape[2], 10)  # check number horizontal patches 
        self.assertEqual(features.shape[3], 1280)  # check number horizontal patches 

        # 2. Re-run with existing output & ensure equal 
        output = pipeline.run(
            name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        reload_features = output['features']

        self.assertTrue(np.array_equal(features, reload_features))

    def test_deep_tissue_feature_extractor(self):
        """
        Test Deep Tissue feature extractor. 
        """

        # 1. load the image
        image = np.array(Image.open(os.path.join(self.image_path, self.image_name)))

        # 2. run deep tissue feature extraction
        feature_extractor = AugmentedDeepTissueFeatureExtractor(
            architecture='mobilenet_v2',
            downsample_factor=2
        )
        features = feature_extractor.process(image)

        self.assertTrue(isinstance(features, torch.Tensor))  # check type 
        self.assertEqual(features.shape[0], 1)  # check number of augmentations (only 1 expected)
        self.assertEqual(features.shape[1], 10)  # check number horizontal patches 
        self.assertEqual(features.shape[2], 10)  # check number horizontal patches 
        self.assertEqual(features.shape[3], 1280)  # check number horizontal patches 

    def test_feature_merger_with_pipeline_runner(self):
        """
        Test feature merger with pipeline runner.
        """

        with open('config/feature_merger.yml', 'r') as file:
            config = yaml.load(file)

        pipeline = PipelineRunner(output_path=self.out_path, save=True, **config)
        pipeline.precompute()
        output = pipeline.run(
            name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        features = output['features']
        self.assertTrue(isinstance(features, torch.Tensor))  # check type 
        self.assertEqual(features.shape[0], 9)  # check number of tissue instances
        self.assertEqual(features.shape[1], 1)  # check number of augmentations
        self.assertEqual(features.shape[2], 1280)  # check number horizontal patches 

        # 2. Re-run with existing output & ensure equal 
        output = pipeline.run(
            name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        reload_features = output['features']

        self.assertTrue(np.array_equal(features, reload_features))

    def test_feature_merger_extractor(self):
        """
        Test Feature Merger extractor. 
        """

        # 1. load the image
        image = np.array(Image.open(os.path.join(self.image_path, self.image_name)))

        # 2. run SLIC extraction
        nr_superpixels = 100
        slic_extractor = ColorMergedSuperpixelExtractor(
            downsampling_factor=4,
            nr_superpixels=nr_superpixels
        )
        superpixels, _ = slic_extractor.process(image)

        # 3. extract deep features
        feature_extractor = AugmentedDeepTissueFeatureExtractor(
            architecture='mobilenet_v2',
            downsample_factor=2
        )
        raw_features = feature_extractor.process(image)

        # 4. feature merger 
        feature_merger = FeatureMerger()
        features = feature_merger.process(raw_features, superpixels)

        self.assertTrue(isinstance(features, torch.Tensor))  # check type 
        self.assertEqual(features.shape[0], 9)  # check number of tissue instances
        self.assertEqual(features.shape[1], 1)  # check number of augmentations
        self.assertEqual(features.shape[2], 1280)  # check number horizontal patches 

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":

    unittest.main()
