"""Unit test for preprocessing.graph_builders"""
import unittest
import numpy as np
import cv2 
import torch 
import yaml
import os 
from dgl.data.utils import save_graphs

from histocartography.preprocessing.nuclei_extraction import NucleiExtractor
from histocartography.preprocessing.feature_extraction import AverageFeatureMerger, DeepFeatureExtractor
from histocartography.preprocessing.graph_builders import RAGGraphBuilder
from histocartography.preprocessing.superpixel import ColorMergedSuperpixelExtractor
from histocartography.visualisation.graph_visualization import GraphVisualization
from histocartography.utils.io import load_image

BASE_N_SEGMENTS = 1000
BASE_N_PIXELS = 100000
MAX_N_SEGMENTS = 10000


class TissueGraphBuildingTestCase(unittest.TestCase):
    """CellGraphBuildingTestCase class."""

    def setUp(self):
        """Setting up the test."""
    
    def test_tissue_graph_building(self):
        """Test cell graph building. Include nuclei detection,
           nuclei labeling and kNN topology construction.
        """

        # 1. load an image
        base_path = '../data'
        image_name = '1238_adh_10.png'
        image = np.array(load_image(os.path.join(base_path, image_name)))

        # 2. super pixel detection 
        nr_superpixels = min(int(BASE_N_SEGMENTS * (image.shape[0] * image.shape[1]/BASE_N_PIXELS)), MAX_N_SEGMENTS)
        print('PUS number of super pixels:', nr_superpixels)
        nr_superpixels = 200
        superpixel_detector = ColorMergedSuperpixelExtractor(
            nr_superpixels=nr_superpixels,
            downsampling_factor=8,
            compactness=20,
            blur_kernel_size=1
        )
        merged_superpixels, superpixels, mapping = superpixel_detector.process(image)

        # 3. super pixel feature extraction 
        feature_extractor = DeepFeatureExtractor(architecture='resnet34', size=96)
        features = feature_extractor.process(image, superpixels)

        # 4. super pixel feature merging
        feature_merger = AverageFeatureMerger()
        merged_features = feature_merger.process(features, mapping)

        # 5. build the tissue graph
        tissue_graph_builder = RAGGraphBuilder()
        tissue_graph = tissue_graph_builder.process(
            structure=merged_superpixels,
            features=merged_features,
        )

        # 6. print graph properties
        print('Number of nodes:', tissue_graph.number_of_nodes())
        print('Number of edges:', tissue_graph.number_of_edges())
        print('Node features:', tissue_graph.ndata['feat'].shape)
        print('Node centroids:', tissue_graph.ndata['centroid'].shape)

        # 7. save DGL graph
        tg_fname = image_name.replace('.png', '_tg.bin')
        save_graphs(os.path.join(base_path, tg_fname), [tissue_graph], labels={"glabel": torch.tensor([0])})

        # 8. visualize the graph 
        visualiser = GraphVisualization(
            show_centroid=True,
            show_edges=False
        )
        out = visualiser.process(
            image=image,
            graph=tissue_graph,
            instance_map=merged_superpixels
        )
        tg_fname = image_name.replace('.png', '_tg.png')
        out.save(os.path.join(base_path, tg_fname))

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    model = TissueGraphBuildingTestCase()
    model.test_tissue_graph_building()

