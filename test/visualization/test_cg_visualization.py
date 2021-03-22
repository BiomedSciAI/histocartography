"""Unit test for visualisation.graph_visualization"""
import unittest
import numpy as np
import cv2
import torch
import yaml
import os
from copy import deepcopy
from dgl.data.utils import load_graphs

from histocartography.utils.io import load_image, save_image
from histocartography.interpretability.grad_cam import GraphGradCAMPPExplainer
from histocartography.visualisation.graph_visualization import (
    GraphVisualization,
    OverlayGraphVisualization,
)

BASE_S3 = "s3://mlflow/"


class GraphVizTestCase(unittest.TestCase):
    """GraphVizTestCase class."""

    def setUp(self):
        """Setting up the test."""

    def test_overlay_with_explanation(self):
        """Test Graph visualization with new class."""
        # 1. load a cell graph
        cell_graph, _ = load_graphs("test/data/cell_graphs/283_dcis_4.bin")
        cell_graph = cell_graph[0]

        # 2. load the corresponding image
        image = np.array(load_image("test/data/images/283_dcis_4.png"))

        # 3. run the explainer
        explainer = GraphGradCAMPPExplainer(model_path="test/data/models/cg_3_classes")
        importance_scores, _ = explainer.process(cell_graph)

        node_attributes = {}
        node_attributes["color"] = importance_scores

        edge_attributes = {}
        edge_attributes["thickness"] = [1, 2, 3]
        edge_attributes["color"] = [0.1, 0.2, 0.8, 0.1, 0.2, 0.3, 0.1, 0.1]

        # 4. run the visualization
        pseudo_instance_map = np.zeros_like(image)
        pseudo_instance_map = pseudo_instance_map[:,:,0]
        pseudo_instance_map[0:50,:]=pseudo_instance_map[0:50,:] + 1
        pseudo_instance_map[:,0:50,]=pseudo_instance_map[:,0:50] + 1
        pseudo_instance_map = pseudo_instance_map + 1
        visualizer = OverlayGraphVisualization(node_style="fill")
        out = visualizer.process(
            image,
            cell_graph,
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
            instance_map=pseudo_instance_map
        )

        # 5. save output image
        save_image("tmp/cg_viz_with_explanation.png", out)

    def test_overlay_graph_viz(self):
        """Test Graph visualization with new class."""
        # 1. load a cell graph
        cell_graph, _ = load_graphs("test/data/cell_graphs/283_dcis_4.bin")
        cell_graph = cell_graph[0]

        # 2. load the corresponding image
        image = np.array(load_image("test/data/images/283_dcis_4.png"))

        # 3. run the visualization
        visualizer = OverlayGraphVisualization()
        out = visualizer.process(image, cell_graph)

        # 5. save output image
        save_image("tmp/cg_viz_overlay.png", out)

    def tearDown(self):
        """Tear down the tests."""


# if __name__ == "__main__":
#     unittest.main()
