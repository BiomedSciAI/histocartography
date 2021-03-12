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
        self.data_path = os.path.join('..', 'data')
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '16B0001851_Block_Region_3.jpg'
        self.out_path = os.path.join(self.data_path, 'stain_normalization_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path) 
        os.makedirs(self.out_path)

    def test_image_loader_with_pipeline_runner(self):
        """
        Test Image Loader with pipeline runner.
        """

        with open('config/image_loader.yml', 'r') as file:
            config = yaml.load(file)
        pipeline = PipelineRunner(output_path=self.out_path, save=False, **config)
        pipeline.precompute()
        output = pipeline.run(
            name=self.image_name.replace('.jpg', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        image = output['image']

        self.assertTrue(isinstance(image, np.ndarray))        # output is numpy 
        self.assertEqual(list(image.shape), [1024, 1280, 3])  # image HxW = mask HxW 

    def test_graph_loader_with_pipeline_runner(self):
        """
        Test DGLGraph Loader with pipeline runner.
        """

        with open('config/graph_loader.yml', 'r') as file:
            config = yaml.load(file)
        pipeline = PipelineRunner(output_path=self.out_path, save=False, **config)
        pipeline.precompute()
        output = pipeline.run(
            name=self.graph_name.replace('.jpg', ''),
            graph_path=os.path.join(self.graph_path, self.graph_name)
        )
        graph = output['graph']

        self.assertTrue(isinstance(graph, dgl.DGLGraph)) # graph is DGLGraph 
        self.assertEqual(graph.number_of_nodes(), 19)    # check number of nodes
        self.assertEqual(graph.number_of_edges(), 56)    # check number of nodes
        self.assertTrue('centroid' in graph.ndata.keys())  # check if centroids
        self.assertTrue('feat' in graph.ndata.keys())      # check if features 

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
