"""Unit test for preprocessing.graph_builders"""
import unittest
import numpy as np
import yaml
import os
import torch
from PIL import Image
import shutil
import dgl

from histocartography import PipelineRunner
from histocartography.preprocessing import DeepFeatureExtractor
from histocartography.preprocessing import AugmentedDeepFeatureExtractor
from histocartography.preprocessing import ColorMergedSuperpixelExtractor
from histocartography.preprocessing import RAGGraphBuilder
from histocartography.preprocessing import KNNGraphBuilder, NucleiExtractor
from histocartography.preprocessing import H5Loader
from histocartography.utils import download_test_data


class GraphBuilderTestCase(unittest.TestCase):
    """GraphBuilderTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        download_test_data(self.data_path)
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '283_dcis_4.png'
        self.nuclei_map_path = os.path.join(self.data_path, 'nuclei_maps')
        self.nuclei_map_name = '283_dcis_4.h5'
        self.annotation_name = '283_dcis_4_annotation.png'
        self.out_path = os.path.join(self.data_path, 'graph_builder_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)

    def test_rag_builder_with_pipeline_runner(self):
        """
        Test rag builder with pipeline runner.
        """

        config_fname = os.path.join(
            self.current_path,
            'config',
            'graph_builder',
            'rag_graph_builder.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        graph = output['graph']

        self.assertTrue(isinstance(graph, dgl.DGLGraph))  # check type
        self.assertEqual(graph.number_of_nodes(), 23)  # check number of nodes
        self.assertEqual(graph.number_of_edges(), 112)  # check number of edges
        # check assigned features
        self.assertEqual(graph.ndata["feat"].shape, (23, 1, 1280))
        # check assigned centroid
        self.assertEqual(graph.ndata["centroid"].shape, (23, 2))

    def test_rag_builder_augmentation_with_pipeline_runner(self):
        """
        Test rag builder with augmentation and pipeline runner.
        """

        config_fname = os.path.join(
            self.current_path,
            'config',
            'graph_builder',
            'rag_graph_builder_aug.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', '_aug'),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        graph = output['graph']

        self.assertTrue(isinstance(graph, dgl.DGLGraph))  # check type
        self.assertEqual(graph.number_of_nodes(), 23)  # check number of nodes
        self.assertEqual(graph.number_of_edges(), 112)  # check number of edges
        # check assigned features
        self.assertEqual(graph.ndata["feat"].shape, (23, 4, 1282))
        # check assigned centroid
        self.assertEqual(graph.ndata["centroid"].shape, (23, 2))

    def test_rag_builder_annotation_with_pipeline_runner(self):
        """
        Test rag builder with annotation and pipeline runner.
        """

        config_fname = os.path.join(
            self.current_path,
            'config',
            'graph_builder',
            'rag_graph_builder_annotation.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', '_annotation'),
            image_path=os.path.join(self.image_path, self.image_name),
            annotation_path=os.path.join(self.image_path, self.annotation_name)
        )
        graph = output['graph']
        labels, counts = np.unique(
            graph.ndata['label'].numpy(), return_counts=True)

        self.assertTrue(isinstance(graph, dgl.DGLGraph))  # check type
        self.assertEqual(graph.number_of_nodes(), 23)  # check number of nodes
        self.assertEqual(graph.number_of_edges(), 112)  # check number of edges
        # check assigned features
        self.assertEqual(graph.ndata["feat"].shape, (23, 1, 1282))
        # check assigned centroid
        self.assertEqual(graph.ndata["centroid"].shape, (23, 2))
        # check assigned node labels
        self.assertTrue(np.array_equal(labels, np.array([0, 1, 2, 4])))
        # check assigned node label counts
        self.assertTrue(np.array_equal(counts, np.array([2, 9, 2, 10])))

    def test_rag_builder(self):
        """
        Test RAG builder.
        """

        # 1. load the image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))

        # 2. run superpixel and merging extraction
        slic_extractor = ColorMergedSuperpixelExtractor(
            downsampling_factor=8,
            superpixel_size=100
        )
        superpixels, _ = slic_extractor.process(image)

        # 3. extract deep features
        feature_extractor = DeepFeatureExtractor(
            architecture='mobilenet_v2',
            patch_size=224,
            downsample_factor=2
        )
        features = feature_extractor.process(input_image=image,
                                             instance_map=superpixels)

        # 4. build the RAG
        rag_builder = RAGGraphBuilder()
        graph = rag_builder.process(superpixels, features)

        self.assertTrue(isinstance(graph, dgl.DGLGraph))  # check type
        self.assertEqual(graph.number_of_nodes(), 23)  # check number of nodes
        self.assertEqual(graph.number_of_edges(), 112)  # check number of edges

    def test_knn_builder_with_pipeline_runner(self):
        """
        Test knn builder with pipeline runner.
        """

        config_fname = os.path.join(
            self.current_path,
            'config',
            'graph_builder',
            'knn_graph_builder.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            output_name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name),
            nuclei_map_path=os.path.join(self.nuclei_map_path, self.nuclei_map_name)
        )
        graph = output['graph']

        self.assertTrue(isinstance(graph, dgl.DGLGraph))  # check type
        self.assertEqual(graph.number_of_nodes(), 331)  # check number of nodes
        self.assertEqual(
            graph.number_of_edges(),
            1655)  # check number of edges

    def test_knn_builder(self):
        """
        Test KNN builder.
        """

        # 1. load the image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))

        # 2. load nuclei detection
        h5_loader = H5Loader()
        instance_map, instance_centroids = h5_loader._process(
            path=os.path.join(
                self.nuclei_map_path,
                self.nuclei_map_name
            )
        )

        # 3. extract deep features
        feature_extractor = AugmentedDeepFeatureExtractor(
            patch_size=224,
            architecture='mobilenet_v2',
            downsample_factor=1
        )
        features = feature_extractor.process(image, instance_map)

        # 5. build the RAG
        knn_builder = KNNGraphBuilder()
        graph = knn_builder.process(instance_map, features)

        self.assertTrue(isinstance(graph, dgl.DGLGraph))  # check type
        self.assertEqual(graph.number_of_nodes(), 331)  # check number of nodes
        self.assertEqual(
            graph.number_of_edges(),
            1655)  # check number of edges

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":

    unittest.main()
