"""Unit test for visualisation.graph_visualization"""
import os
import shutil
import unittest
import numpy as np
from dgl.data.utils import load_graphs
from PIL import Image

from histocartography.preprocessing import (
    DeepFeatureExtractor,
    AugmentedDeepFeatureExtractor,
    RAGGraphBuilder,
)
from histocartography.preprocessing.nuclei_extraction import NucleiExtractor
from histocartography.preprocessing.superpixel import SLICSuperpixelExtractor
from histocartography.utils import download_test_data
from histocartography.visualization.visualization import (
    HACTVisualization,
    InstanceImageVisualization,
    OverlayGraphVisualization,
)


class GraphVizTestCase(unittest.TestCase):
    """GraphVizTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, "..", "data")
        download_test_data(self.data_path)
        self.image_path = os.path.join(self.data_path, "images")
        self.image_name = "283_dcis_4.png"
        self.cell_graph_path = os.path.join(self.data_path, "cell_graphs")
        self.graph_name = "283_dcis_4.bin"
        self.tissue_graph_path = os.path.join(self.data_path, "tissue_graphs")
        self.tissue_graph_name = "283_dcis_4.bin"
        self.out_path = os.path.join(self.data_path, "visualization_test")
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)

    def test_overlay_with_explanation(self):
        """Test Graph visualization with explanation."""

        # 1. load a cell graph
        cell_graph, _ = load_graphs(os.path.join(
            self.cell_graph_path, self.graph_name))
        cell_graph = cell_graph[0]

        # 2. load the corresponding image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))

        # 3. fake explainer importance scores
        importance_scores = np.random.normal(0.7, 0.1, 100)

        node_attributes = {}
        node_attributes["color"] = importance_scores

        edge_attributes = {}
        edge_attributes["thickness"] = [1, 2, 3]
        edge_attributes["color"] = [0.1, 0.2, 0.8, 0.1, 0.2, 0.3, 0.1, 0.1]

        # 4. run the visualization
        visualizer = OverlayGraphVisualization(node_style="fill")
        out = visualizer.process(
            image,
            cell_graph,
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
        )

        # 5. save output image
        out.save(
            os.path.join(
                self.out_path,
                self.image_name.replace(".png", "") + "_cg_explanation.png",
            ),
            quality=95
        )

    def test_overlay_graph_viz(self):
        """Test Graph visualization with nuclei maps."""

        # 1. load a cell graph
        cell_graph, _ = load_graphs(os.path.join(
            self.cell_graph_path, self.graph_name))
        cell_graph = cell_graph[0]

        # 2. load the corresponding image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))

        # 3. run the visualization
        visualizer = OverlayGraphVisualization(
            instance_visualizer=InstanceImageVisualization(
                instance_style="filled+outline", colormap="jet"
            )
        )
        out = visualizer.process(image, cell_graph)

        # 4. save output image
        out.save(
            os.path.join(
                self.out_path,
                self.image_name.replace(
                    ".png",
                    "") +
                "_cg_overlay.png"),
            quality=95)

    def test_superpixel_viz(self):
        """Test Nuclei visualization."""

        # 1. load the corresponding image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))

        # 2. extract nuclei
        extractor = SLICSuperpixelExtractor(nr_superpixels=50)
        instance_map = extractor.process(image)

        # 3. run the visualization
        visualizer = InstanceImageVisualization(color="darkgreen")
        out = visualizer.process(image, instance_map=instance_map)

        # 5. save output image
        out.save(
            os.path.join(
                self.out_path,
                self.image_name.replace(
                    ".png",
                    "") +
                "_superpixel_overlay.png",
            ),
            quality=95)

    def test_hact_viz(self):
        """Test hierarchical visualization."""

        # 1. load the corresponding image
        image = np.array(
            Image.open(
                os.path.join(
                    self.image_path,
                    self.image_name)))

        # 2. load tissue graph
        tissue_graph, _ = load_graphs(os.path.join(
            self.tissue_graph_path, self.graph_name))
        tissue_graph = tissue_graph[0]

        # 3. load cell graph
        cell_graph, _ = load_graphs(os.path.join(
            self.cell_graph_path, self.graph_name))
        cell_graph = cell_graph[0]

        # 6. run the visualization
        visualizer = HACTVisualization()
        out = visualizer.process(
            image,
            cell_graph=cell_graph,
            tissue_graph=tissue_graph,
        )

        # 5. save output image
        out.save(
            os.path.join(
                self.out_path,
                self.image_name.replace(
                    ".png",
                    "") +
                "_hierarchical_overlay.png",
            ),
            quality=95)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    unittest.main()
