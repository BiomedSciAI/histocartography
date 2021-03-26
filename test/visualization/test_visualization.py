"""Unit test for visualisation.graph_visualization"""
import os
import shutil
import unittest

import numpy as np
from dgl.data.utils import load_graphs
from histocartography.preprocessing import (
    AugmentedDeepTissueFeatureExtractor,
    FeatureMerger,
    RAGGraphBuilder,
)
from histocartography.preprocessing.nuclei_extraction import NucleiExtractor
from histocartography.preprocessing.superpixel import SLICSuperpixelExtractor
from histocartography.utils.io import load_image, save_image
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
        self.image_path = os.path.join(self.data_path, "images")
        self.image_name = "283_dcis_4.png"
        self.graph_path = os.path.join(self.data_path, "cell_graphs")
        self.graph_name = "283_dcis_4.bin"
        self.tissue_graph_path = os.path.join(self.data_path, "tissue_graphs")
        self.tissue_graph_name = "283_dcis_4_tg.bin"
        self.out_path = os.path.join(self.data_path, "visualization_test")
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)

    def test_overlay_with_explanation(self):
        """Test Graph visualization with explanation."""

        # 1. load a cell graph
        cell_graph, _ = load_graphs(os.path.join(self.graph_path, self.graph_name))
        cell_graph = cell_graph[0]

        # 2. load the corresponding image
        image = np.array(load_image(os.path.join(self.image_path, self.image_name)))

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
        save_image(
            os.path.join(
                self.out_path,
                self.image_name.replace(".png", "") + "_cg_explanation.png",
            ),
            out,
        )

    def test_overlay_graph_viz(self):
        """Test Graph visualization with nuclei maps."""

        # 1. load a cell graph
        cell_graph, _ = load_graphs(os.path.join(self.graph_path, self.graph_name))
        cell_graph = cell_graph[0]

        # 2. load the corresponding image
        image = np.array(load_image(os.path.join(self.image_path, self.image_name)))
        # 3. detect nuclei (for visualization)
        extractor = NucleiExtractor(pretrained_data="monusac")
        nuclei, nuclei_centroids = extractor.process(image)

        # 3. run the visualization
        visualizer = OverlayGraphVisualization(
            instance_visualizer=InstanceImageVisualization(
                instance_style="filled+outline", colormap="jet"
            )
        )
        out = visualizer.process(image, cell_graph, instance_map=nuclei)

        # 5. save output image
        save_image(
            os.path.join(
                self.out_path, self.image_name.replace(".png", "") + "_cg_overlay.png"
            ),
            out,
        )

    def test_superpixel_viz(self):
        """Test Nuclei visualization."""

        # 1. load the corresponding image
        image = np.array(load_image(os.path.join(self.image_path, self.image_name)))

        # 2. extract nuclei
        extractor = SLICSuperpixelExtractor(50)
        instance_map = extractor.process(image)

        # 3. run the visualization
        visualizer = InstanceImageVisualization(color="darkgreen")
        out = visualizer.process(image, instance_map=instance_map)

        # 5. save output image
        save_image(
            os.path.join(
                self.out_path,
                self.image_name.replace(".png", "") + "_superpixel_overlay.png",
            ),
            out,
        )

    def test_hact_viz(self):
        """Test hierarchical visualization."""

        # 1. load the corresponding image
        image = np.array(load_image(os.path.join(self.image_path, self.image_name)))

        # 2. extract instances
        extractor = SLICSuperpixelExtractor(50)
        tissue_instance_map = extractor.process(image)

        # 3. load graphs
        cell_graph, _ = load_graphs(os.path.join(self.graph_path, self.graph_name))
        cell_graph = cell_graph[0]

        feature_extractor = AugmentedDeepTissueFeatureExtractor(
            architecture="mobilenet_v2", downsample_factor=2
        )
        raw_features = feature_extractor.process(image)

        # 4. feature merger
        feature_merger = FeatureMerger()
        features = feature_merger.process(raw_features, tissue_instance_map)

        rag_builder = RAGGraphBuilder()
        tissue_graph = rag_builder.process(tissue_instance_map, features)
        # tissue_graph, _ = load_graphs(os.path.join(self.tissue_graph_path, self.tissue_graph_name))
        # tissue_graph = tissue_graph[0]

        # 4. run the visualization
        visualizer = HACTVisualization()
        out = visualizer.process(
            image,
            cell_graph=cell_graph,
            tissue_graph=tissue_graph,
            tissue_instance_map=tissue_instance_map,
        )

        # 5. save output image
        save_image(
            os.path.join(
                self.out_path,
                self.image_name.replace(".png", "") + "_hierarchical_overlay.png",
            ),
            out,
        )

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    unittest.main()
