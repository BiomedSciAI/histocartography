"""Unit test for preprocessing.graph_builders"""
import unittest
import numpy as np
import cv2 
import torch 
import yaml
import os 
from dgl.data.utils import save_graphs

from histocartography.preprocessing.pipeline import PipelineRunner
from histocartography.preprocessing.nuclei_extraction import NucleiExtractor
from histocartography.preprocessing.feature_extraction import DeepFeatureExtractor
from histocartography.preprocessing.graph_builders import KNNGraphBuilder
from histocartography.utils.io import load_image
from histocartography.utils.hover import visualize_instances


NR_NUCLEI_TYPES = 6

class CellGraphBuildingTestCase(unittest.TestCase):
    """CellGraphBuildingTestCase class."""

    def setUp(self):
        """Setting up the test."""
    
    # def test_cell_graph_building_with_pipeline_runner(self):
    #     """Test cell graph building with pipeline runner. Include nuclei detection,
    #        nuclei labeling and kNN topology construction.
    #     """

    #     with open('config/dummy.yml', 'r') as file:
    #         config = yaml.load(file)
    #     pipeline = PipelineRunner(output_path='output', save=False, **config)
    #     pipeline.precompute()
    #     output = pipeline.run(name="CG_TEST", image_path='../data/1937_benign_4.png')

    #     # 5. print graph properties
    #     cell_graph = output['graph']
    #     print('Number of nodes:', cell_graph.number_of_nodes())
    #     print('Number of edges:', cell_graph.number_of_edges())
    #     print('Node features:', cell_graph.ndata['feat'].shape)
    #     print('Node centroids:', cell_graph.ndata['centroid'].shape)

    def test_cell_graph_building(self):
        """Test cell graph building. Include nuclei detection,
           nuclei labeling and kNN topology construction.
        """

        base_path = '../data'
        image_fnames = ['1238_adh_10.png', '1286_udh_35.png', '1937_benign_4.png', '283_dcis_4.png', '311_fea_25.png']

        for image_name in image_fnames:

            print('*** Testing image {}'.format(image_name))

            # 1. load the image
            image = np.array(load_image(os.path.join(base_path, 'images', image_name)))

            # 2. nuclei detection 
            nuclei_detector = NucleiExtractor(
                model_path='checkpoints/hovernet_kumar_notype.pth'
            )
            instance_map, instance_centroids = nuclei_detector.process(image)

            # 3. nuclei feature extraction 
            nuclei_feature_extractor = DeepFeatureExtractor(
                architecture='resnet34',
                size=72
            )
            deep_features = nuclei_feature_extractor.process(image, instance_map)
            position_features = torch.FloatTensor(instance_centroids / image.shape[:-1])
            instance_features = torch.cat((deep_features, position_features), dim=1)

            # 4. build the cell graph
            cell_graph_builder = KNNGraphBuilder(
                k=5,
                thresh=50,
                nr_classes=NR_NUCLEI_TYPES
            )
            cell_graph = cell_graph_builder.process(
                structure=instance_centroids,
                features=instance_features,
            )

            # 5. print graph properties
            print('Number of nodes:', cell_graph.number_of_nodes())
            print('Number of edges:', cell_graph.number_of_edges())
            print('Node features:', cell_graph.ndata['feat'].shape)
            print('Node centroids:', cell_graph.ndata['centroid'].shape)

            # 6. save as DGL graph
            cg_fname = image_name.replace('.png', '_cg.bin')
            save_graphs(os.path.join(base_path, 'cell_graphs', cg_fname), [cell_graph])

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    model = CellGraphBuildingTestCase()
    model.test_cell_graph_building()
    # model.test_cell_graph_building_with_pipeline_runner()

