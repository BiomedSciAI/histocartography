"""Unit test for visualisation.graph_visualization"""
import unittest
import numpy as np
import cv2 
import torch 
import yaml
from copy import deepcopy
from dgl.data.utils import load_graphs

from histocartography.utils.io import load_image, save_image
from histocartography.interpretability.saliency_explainer.graph_gradcam_explainer import GraphGradCAMExplainer
from histocartography.visualisation.graph_visualization import GraphVisualization

BASE_S3 = 's3://mlflow/'


class GraphVizTestCase(unittest.TestCase):
    """GraphVizTestCase class."""

    def setUp(self):
        """Setting up the test."""
    
    def test_graph_viz_with_importance_score(self):
        """Test Graph visualization with importance scores.
        """

        # 1. load a cell graph
        cell_graph, _ = load_graphs('../data/283_dcis_4.bin')
        cell_graph = cell_graph[0]

        # 2. load the corresponding image
        image = np.array(load_image('../data/283_dcis_4.png'))

        # 3. run the explainer
        explainer = GraphGradCAMExplainer(
            model_path=BASE_S3 + '29b7f5ee991e4a3e8b553b49a1c3c05a/artifacts/model_best_val_weighted_f1_score_0'
        )
        importance_scores, _ = explainer.process(cell_graph)

        # 4. run the visualization
        visualizer = GraphVisualization()
        out = visualizer.process(image, cell_graph, node_importance=importance_scores)
         
        # 5. save output image 
        save_image('cg_viz_with_explanation.png', out)

    def test_graph_viz_without_importance_score(self):
        """Test Graph visualization with importance scores.
        """

        # 1. load a cell graph
        cell_graph, _ = load_graphs('../data/283_dcis_4.bin')
        cell_graph = cell_graph[0]

        # 2. load the corresponding image
        image = np.array(load_image('../data/283_dcis_4.png'))

        # 3. run the visualization
        visualizer = GraphVisualization(show_edges=True, centroid_fill=None)
        out = visualizer.process(image, cell_graph)
         
        # 5. save output image 
        save_image('cg_viz.png', out)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    model = GraphVizTestCase()
    model.test_graph_viz_with_importance_score()
    model.test_graph_viz_without_importance_score()
