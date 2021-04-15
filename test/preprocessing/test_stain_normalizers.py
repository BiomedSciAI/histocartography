"""Unit test for preprocessing.stain_normalizers"""
import unittest
import numpy as np
import yaml
import os
from PIL import Image
import shutil

from histocartography import PipelineRunner
from histocartography.preprocessing import VahadaneStainNormalizer, MacenkoStainNormalizer
from histocartography.utils import download_test_data


class StainNormalizationTestCase(unittest.TestCase):
    """StainNormalizationTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        download_test_data(self.data_path)
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '17B0031061.png'
        self.out_path = os.path.join(
            self.data_path, 'stain_normalization_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)

    def test_macenko_normalizer_no_ref(self):
        """
        Test Macenko Stain Normalization: without Reference.
        """

        config_fname = os.path.join(
            self.current_path,
            'config',
            'stain_normalization',
            'macenko_normalizer_noref.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))
        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        image_norm = output['normalized_image']

        self.assertTrue(isinstance(image_norm, np.ndarray))  # output is numpy
        self.assertEqual(
            list(
                image.shape), list(
                image_norm.shape))  # image HxW = mask HxW

    def test_macenko_normalizer_ref(self):
        """
        Test Macenko Stain Normalization: with Reference.
        """

        config_fname = os.path.join(
            self.current_path,
            'config',
            'stain_normalization',
            'macenko_normalizer_ref.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)
        config['stages'][1]['preprocessing']['params']['target_path'] = os.path.join(
            self.current_path, config['stages'][1]['preprocessing']['params']['target_path'])

        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))
        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        image_norm = output['normalized_image']

        self.assertTrue(isinstance(image_norm, np.ndarray))  # output is numpy
        self.assertEqual(
            list(
                image.shape), list(
                image_norm.shape))  # image HxW = mask HxW

    def test_declarative_macenko_normalizer_ref(self):
        """
        Test Macenko Stain Normalization with Reference and declarative syntax.
        """

        normalizer = MacenkoStainNormalizer(target_path=os.path.join(
            self.current_path,
            '..',
            'data',
            'images',
            '18B000646H.png')
        )
        # load image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))
        image_norm = normalizer.process(image)

        self.assertTrue(isinstance(image_norm, np.ndarray))  # output is numpy
        self.assertEqual(
            list(
                image.shape), list(
                image_norm.shape))  # image HxW = mask HxW

    def test_declarative_macenko_normalizer_noref(self):
        """
        Test Macenko Stain Normalization without Reference and declarative syntax.
        """

        normalizer = MacenkoStainNormalizer()

        # load image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))
        image_norm = normalizer.process(image)

        self.assertTrue(isinstance(image_norm, np.ndarray))  # output is numpy
        self.assertEqual(
            list(
                image.shape), list(
                image_norm.shape))  # image HxW = mask HxW

    def test_vahadane_normalizer_no_ref(self):
        """
        Test Vahadane Stain Normalization: without Reference.
        """

        config_fname = os.path.join(
            self.current_path,
            'config',
            'stain_normalization',
            'vahadane_normalizer_noref.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))
        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        image_norm = output['normalized_image']

        self.assertTrue(isinstance(image_norm, np.ndarray))  # output is numpy
        self.assertEqual(
            list(
                image.shape), list(
                image_norm.shape))  # image HxW = mask HxW

    def test_vahadane_normalizer_ref(self):
        """
        Test Vahadane Stain Normalization: with Reference.
        """

        config_fname = os.path.join(
            self.current_path,
            'config',
            'stain_normalization',
            'vahadane_normalizer_ref.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)
        config['stages'][1]['preprocessing']['params']['target_path'] = os.path.join(
            self.current_path, config['stages'][1]['preprocessing']['params']['target_path'])

        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))
        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        image_norm = output['normalized_image']

        self.assertTrue(isinstance(image_norm, np.ndarray))  # output is numpy
        self.assertEqual(
            list(
                image.shape), list(
                image_norm.shape))  # image HxW = mask HxW

    def test_declarative_vahadane_normalizer_ref(self):
        """
        Test Vahadane Stain Normalization without Reference and declarative syntax.
        """

        normalizer = VahadaneStainNormalizer(target_path=os.path.join(
            self.current_path,
            '..',
            'data',
            'images',
            '18B000646H.png')
        )
        # load image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))
        image_norm = normalizer.process(image)

        self.assertTrue(isinstance(image_norm, np.ndarray))  # output is numpy
        self.assertEqual(
            list(
                image.shape), list(
                image_norm.shape))  # image HxW = mask HxW

    def test_declarative_vahadane_normalizer_noref(self):
        """
        Test Vahadane Stain Normalization with Reference and declarative syntax.
        """

        normalizer = VahadaneStainNormalizer()

        # load image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))
        image_norm = normalizer.process(image)

        self.assertTrue(isinstance(image_norm, np.ndarray))  # output is numpy
        self.assertEqual(
            list(
                image.shape), list(
                image_norm.shape))  # image HxW = mask HxW

    def test_vahadane_invalid_precomputed_normalizer(self):
        """
        Test Vahadane invalid precomputed normalization.
        """

        config_fname = os.path.join(
            self.current_path,
            'config',
            'stain_normalization',
            'vahadane_precomputed_normalizer_fail.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        with self.assertRaises(FileNotFoundError):
            pipeline = PipelineRunner(output_path=None, **config)

    def test_vahadane_target_precomputed_provided(self):
        """
        Test Vahadane when providing both a target image and a path to precomputed.
        """
        with self.assertRaises(AssertionError):
            normalizer = VahadaneStainNormalizer(
                target_path='some_dummy_path',
                precomputed_normalizer_path='some_other_dummy_path'
            )

    def test_macenko_target_precomputed_provided(self):
        """
        Test Macenko when providing both a target image and a path to precomputed.
        """
        with self.assertRaises(AssertionError):
            normalizer = MacenkoStainNormalizer(
                target_path='some_dummy_path',
                precomputed_normalizer_path='some_other_dummy_path'
            )

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    unittest.main()
