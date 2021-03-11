"""Unit test for preprocessing.graph_builders"""
import unittest
import numpy as np
import yaml
import os 
import torch 
from PIL import Image
import shutil
import dgl 

from histocartography.pipeline import PipelineRunner
from histocartography.preprocessing import HandcraftedFeatureExtractor, DeepTissueFeatureExtractor
from histocartography.preprocessing import AugmentedDeepTissueFeatureExtractor, FeatureMerger
from histocartography.preprocessing import ColorMergedSuperpixelExtractor, RAGGraphBuilder


class GraphBuilderTestCase(unittest.TestCase):
    """GraphBuilderTestCase class."""

    @classmethod
    def setUpClass(self):
        self.data_path = os.path.join('..', 'data')
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '283_dcis_4.png'
        self.out_path = os.path.join(self.data_path, 'graph_builder_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path) 
        os.makedirs(self.out_path)

    # def test_rag_builder_with_pipeline_runner(self):
    #     """
    #     Test rag builder with pipeline runner.
    #     """

    #     with open('config/rag_graph_builder.yml', 'r') as file:
    #         config = yaml.load(file)

    #     pipeline = PipelineRunner(output_path=self.out_path, save=True, **config)
    #     pipeline.precompute()
    #     output = pipeline.run(
    #         name=self.image_name.replace('.png', ''),
    #         image_path=os.path.join(self.image_path, self.image_name)
    #     )
    #     graph = output['graph']

    #     self.assertTrue(isinstance(graph, dgl.DGLGraph))  # check type 
    #     self.assertEqual(graph.number_of_nodes(), 9)  # check number of tissue instances
    #     self.assertEqual(graph.number_of_edges(), 24)  # check number of augmentations

    def test_rag_builder(self):
        """
        Test RAG builder. 
        """

        # 1. load the image
        image = np.array(Image.open(os.path.join(self.image_path, self.image_name)))

        # 2. run superpixel and merging extraction
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

        # 5. build the RAG
        rag_builder = RAGGraphBuilder()
        graph = rag_builder.process(superpixels, features)

        self.assertTrue(isinstance(graph, dgl.DGLGraph))  # check type 
        self.assertEqual(graph.number_of_nodes(), 9)  # check number of tissue instances
        self.assertEqual(graph.number_of_edges(), 24)  # check number of augmentations

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":

    unittest.main()
