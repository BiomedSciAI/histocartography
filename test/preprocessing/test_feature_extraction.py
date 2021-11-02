"""Unit test for preprocessing.feature_extraction"""
import unittest
import numpy as np
import pandas as pd
import yaml
import os
import torch
import shutil

from histocartography import PipelineRunner
from histocartography.utils import download_test_data


class FeatureExtractionTestCase(unittest.TestCase):
    """FeatureExtractionTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        download_test_data(self.data_path)
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '283_dcis_4.png'
        self.bg_image_name = '283_dcis_4_background.png'
        self.nuclei_map_path = os.path.join(self.data_path, 'nuclei_maps')
        self.nuclei_map_name = '283_dcis_4.h5'
        self.out_path = os.path.join(self.data_path, 'feature_extraction_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)

    def test_handcrafted_feature_extractor(self):
        """
        Test handcrafted feature extractor with pipeline runner.
        """
        config_fname = os.path.join(self.current_path,
                                    'config',
                                    'feature_extraction',
                                    'handcrafted_feature_extractor.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        features = output['features']

        self.assertTrue(isinstance(features, torch.Tensor))  # check type
        # check number of features per instance
        self.assertEqual(features.shape[1], 24)
        # check number of instances detected
        self.assertEqual(features.shape[0], 23)

        # Re-run with existing output & ensure equal
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        reload_features = output['features']

        self.assertTrue(np.array_equal(features, reload_features))

    def test_deep_tissue_feature_extractor_noaug(self):
        """
        Test deep tissue feature extractor with pipeline runner and without augmentation.
        """
        config_fname = os.path.join(self.current_path,
                                    'config',
                                    'feature_extraction',
                                    'deep_tissue_feature_extractor_noaug.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        features = output['features']

        self.assertTrue(isinstance(features, torch.Tensor))  # check type
        # check number of superpixels
        self.assertEqual(features.shape[0], 23)
        self.assertEqual(features.shape[1], 1280)  # check number features

        # Re-run with existing output & ensure equal
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        reload_features = output['features']

        self.assertTrue(np.array_equal(features, reload_features))

    def test_deep_tissue_feature_extractor_noaug_masked_instance(self):
        """
        Test deep tissue feature extractor with pipeline runner without augmentation
        and by masking the pixels outside the instance.
        """
        config_fname = os.path.join(self.current_path,
                                    'config',
                                    'feature_extraction',
                                    'deep_tissue_feature_extractor_noaug_masked_instance.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        features = output['features']

        self.assertTrue(isinstance(features, torch.Tensor))  # check type
        # check number of superpixels
        self.assertEqual(features.shape[0], 43)
        self.assertEqual(features.shape[1], 1280)  # check number features

        # Re-run with existing output & ensure equal
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        reload_features = output['features']

        self.assertTrue(np.array_equal(features, reload_features))

    def test_deep_tissue_feature_extractor_aug(self):
        """
        Test deep tissue feature extractor with pipeline runner and with augmentation.
        """
        config_fname = os.path.join(self.current_path,
                                    'config',
                                    'feature_extraction',
                                    'deep_tissue_feature_extractor_aug.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        features = output['features']

        self.assertTrue(isinstance(features, torch.Tensor))  # check type
        # check number of superpixels
        self.assertEqual(features.shape[0], 23)
        # check number of augmentations
        self.assertEqual(features.shape[1], 4)
        self.assertEqual(features.shape[2], 1280)  # check number features

        # Re-run with existing output & ensure equal
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        reload_features = output['features']

        self.assertTrue(np.array_equal(features, reload_features))

    def test_deep_nuclei_feature_extractor_noaug(self):
        """Test deep nuclei feature extractor with pipeline runner and without augmentation."""

        config_fname = os.path.join(self.current_path,
                                    'config',
                                    'feature_extraction',
                                    'deep_nuclei_feature_extractor_noaug.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name),
            nuclei_map_path=os.path.join(self.nuclei_map_path, self.nuclei_map_name)
        )
        features = output['features']

        self.assertTrue(isinstance(features, torch.Tensor))  # check type
        # check number of nuclei
        self.assertEqual(features.shape[0], 331)
        # check number features
        self.assertEqual(features.shape[1], 1280)

    def test_deep_nuclei_feature_extractor_aug(self):
        """Test deep nuclei feature extractor with pipeline runner and with augmentation."""

        config_fname = os.path.join(self.current_path,
                                    'config',
                                    'feature_extraction',
                                    'deep_nuclei_feature_extractor_aug.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name),
            nuclei_map_path=os.path.join(self.nuclei_map_path, self.nuclei_map_name)
        )
        features = output['features']

        self.assertTrue(isinstance(features, torch.Tensor))  # check type
        self.assertEqual(features.shape[0], 331)   # check number of nuclei
        # check number of augmentations
        self.assertEqual(features.shape[1], 4)
        self.assertEqual(features.shape[2], 1280)  # check number features

    def test_grid_deep_tissue_feature_extractor_noaug(self):
        """
        Test grid deep tissue feature extractor with pipeline runner and without augmentation.
        """
        config_fname = os.path.join(
            self.current_path,
            'config',
            'feature_extraction',
            'grid_deep_tissue_feature_extractor_noaug.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        features = output['features']

        self.assertTrue(
            isinstance(
                features,
                torch.Tensor))         # check type
        # check number of dimensions
        self.assertEqual(features.ndim, 3)
        # check rows of feature cube
        self.assertEqual(features.shape[0], 7)
        # check columns of feature cube
        self.assertEqual(features.shape[1], 7)
        # check number features
        self.assertEqual(features.shape[2], 1280)

        # Re-run with existing output & ensure equal
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        reload_features = output['features']

        self.assertTrue(np.array_equal(features, reload_features))

    def test_grid_deep_tissue_feature_extractor_aug(self):
        """
        Test grid deep tissue feature extractor with pipeline runner and with augmentation.
        """
        config_fname = os.path.join(
            self.current_path,
            'config',
            'feature_extraction',
            'grid_deep_tissue_feature_extractor_aug.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        features = output['features']

        self.assertTrue(isinstance(features, torch.Tensor))     # check type
        # check number of dimensions
        self.assertEqual(features.ndim, 4)
        # check rows of feature cube
        self.assertEqual(features.shape[0], 7)
        # check columns of feature cube
        self.assertEqual(features.shape[1], 7)
        # check number of augmentations
        self.assertEqual(features.shape[2], 4)
        # check number features
        self.assertEqual(features.shape[3], 1280)

        # Re-run with existing output & ensure equal
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        reload_features = output['features']

        self.assertTrue(np.array_equal(features, reload_features))

    def test_masked_grid_deep_tissue_feature_extractor(self):
        """
        Test grid deep tissue feature extractor with pipeline runner and without augmentation.
        """
        config_fname = os.path.join(
            self.current_path,
            'config',
            'feature_extraction',
            'masked_grid_deep_tissue_feature_extractor.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.bg_image_name)
        )
        index_filter = output['index_filter']
        features = output['features']
        mask = output['mask']

        # check type
        self.assertTrue(isinstance(index_filter, pd.DataFrame))
        self.assertTrue(isinstance(features, pd.DataFrame))
        # check dimensions
        self.assertEqual(index_filter.shape, (1, 49))
        self.assertEqual(features.shape, (1280, 49))
        self.assertEqual(mask.shape, (717, 717))
        # check that all background patches are invalid
        gt_index_filter = ([False]*7
                           + ([False] + [True]*5 + [False])*4
                           + [False] + [True]*3 + [False]*3
                           + [False]*7)
        self.assertEqual(list(index_filter.loc['is_valid']), gt_index_filter)

        # Re-run with existing output & ensure equal
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.bg_image_name)
        )
        reload_features = output['features']

        self.assertTrue(np.array_equal(features, reload_features))

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":

    unittest.main()
