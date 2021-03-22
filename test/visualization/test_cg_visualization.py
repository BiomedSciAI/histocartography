"""Unit test for visualisation.graph_visualization"""
import unittest
import numpy as np
import yaml
import os
from dgl.data.utils import load_graphs
import shutil

from histocartography.utils.io import load_image, save_image
from histocartography.interpretability.grad_cam import GraphGradCAMPPExplainer
from histocartography.visualisation.graph_visualization import OverlayGraphVisualization


class GraphVizTestCase(unittest.TestCase):
    """GraphVizTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '283_dcis_4.png'
        self.graph_path = os.path.join(self.data_path, 'cell_graphs')
        self.graph_name = '283_dcis_4.bin'
        self.out_path = os.path.join(self.data_path, 'visualization_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path) 
        os.makedirs(self.out_path)

    def test_overlay_with_explanation(self):
        """Test Graph visualization."""

        # 1. load a cell graph
        cell_graph, _ = load_graphs(os.path.join(self.graph_path, self.graph_name))
        cell_graph = cell_graph[0]

        # 2. load the corresponding image
        image = np.array(load_image(os.path.join(self.image_path, self.image_name)))

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
        pseudo_instance_map = pseudo_instance_map[:, :, 0]
        pseudo_instance_map[0:50, :] = pseudo_instance_map[0:50, :] + 1
        pseudo_instance_map[:, 0:50] = (
            pseudo_instance_map[:, 0:50] + 1
        )
        pseudo_instance_map = pseudo_instance_map + 1
        visualizer = OverlayGraphVisualization(node_style="fill")
        out = visualizer.process(
            image,
            cell_graph,
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
            instance_map=pseudo_instance_map,
        )

        # 5. save output image
        save_image(
            os.path.join(
                self.out_path,
                self.image_name.replace('.png', '') + '_cg_explanation.png'),
            out
        )

    def test_overlay_graph_viz(self):
        """Test Graph visualization."""

        # 1. load a cell graph
        cell_graph, _ = load_graphs(os.path.join(self.graph_path, self.graph_name))
        cell_graph = cell_graph[0]

        # 2. load the corresponding image
        image = np.array(load_image(os.path.join(self.image_path, self.image_name)))

        # 3. run the visualization
        visualizer = OverlayGraphVisualization()
        out = visualizer.process(image, cell_graph)

        # 5. save output image
        save_image(
            os.path.join(
                self.out_path,
                self.image_name.replace('.png', '') + '_cg_overlay.png'),
            out
        )
    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    unittest.main()
