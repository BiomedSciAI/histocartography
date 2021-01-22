"""Unit test for preprocessing.graph_builders"""
import unittest
import numpy as np
import cv2 
import torch 
import yaml
from dgl.data.utils import save_graphs

from histocartography.preprocessing.nuclei_extraction import NucleiExtractor
from histocartography.preprocessing.feature_extraction import DeepFeatureExtractor, FeatureMerger
from histocartography.preprocessing.graph_builders import RAGGraphBuilder
from histocartography.preprocessing.superpixel import SLICSuperpixelExtractor, SpecialSuperpixelMerger
from histocartography.visualisation.graph_visualization import GraphVisualization
from histocartography.utils.io import load_image


class TissueGraphBuildingTestCase(unittest.TestCase):
    """CellGraphBuildingTestCase class."""

    def setUp(self):
        """Setting up the test."""
    
    def test_tissue_graph_building(self):
        """Test cell graph building. Include nuclei detection,
           nuclei labeling and kNN topology construction.
        """

        # 1. load an image
        image = np.array(load_image('../data/1607_adh_10.png'))

        # 2. super pixel detection 
        superpixel_detector = SLICSuperpixelExtractor(nr_superpixels=200, downsampling_factor=4)
        superpixels = superpixel_detector.process(image)

        # 3. super pixel feature extraction 
        feature_extractor = DeepFeatureExtractor(architecture='resnet34')
        features = feature_extractor.process(image, superpixels)

        # 4. super pixel merging 
        superpixel_merger = SpecialSuperpixelMerger(downsampling_factor=8)
        merged_superpixels = superpixel_merger.process(image, superpixels)

        # 5. super pixel feature merging
        feature_merger = FeatureMerger(downsampling_factor=8)
        merged_features = feature_merger.process(superpixels, merged_superpixels, features)

        # 4. build the tissue graph
        tissue_graph_builder = RAGGraphBuilder()
        tissue_graph = tissue_graph_builder.process(
            structure=merged_superpixels,
            features=merged_features,
        )

        # 5. print graph properties
        print('Number of nodes:', tissue_graph.number_of_nodes())
        print('Number of edges:', tissue_graph.number_of_edges())
        print('Node features:', tissue_graph.ndata['feat'].shape)
        print('Node centroids:', tissue_graph.ndata['centroid'].shape)

        # 6. save DGL graph
        save_graphs("../data/1607_adh_10_tg.bin", [tissue_graph], labels={"glabel": torch.tensor([1])})

        # 7. visualize the graph 
        visualiser = GraphVisualization(
            show_centroid=True,
            show_edges=True
        )
        out = visualiser.process(
            image=image,
            graph=tissue_graph,
            instance_map=merged_superpixels
            )
        out.save('../data/1607_adh_10_tg_viz.png')

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    model = TissueGraphBuildingTestCase()
    model.test_tissue_graph_building()

