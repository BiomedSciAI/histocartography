"""Unit test for interpretability.gradcam"""
import unittest
import numpy as np
import cv2
import torch
import yaml
from copy import deepcopy
import h5py
import os
import shutil
from dgl.data.utils import load_graphs

from histocartography.interpretability import GraphGradCAMExplainer, GraphGradCAMPPExplainer
from histocartography.ml import CellGraphModel
from histocartography.utils import set_graph_on_cuda, download_test_data

IS_CUDA = torch.cuda.is_available()


class GraphGradCAMTestCase(unittest.TestCase):
    """GraphGradCAMTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        download_test_data(self.data_path)
        self.graph_path = os.path.join(self.data_path, 'tissue_graphs')
        self.graph_name = '283_dcis_4.bin'
        self.out_path = os.path.join(self.data_path, 'graph_graphcam_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)

    def test_graphgradcam_with_saving(self):
        """
        Test Graph GradCAM with saving.
        """

        # 1. load a graph
        graph, _ = load_graphs(os.path.join(self.graph_path, self.graph_name))
        graph = graph[0]
        graph = set_graph_on_cuda(graph) if IS_CUDA else graph
        node_dim = graph.ndata['feat'].shape[1]

        # 2. create model
        config_fname = os.path.join(
            self.current_path,
            'config',
            'cg_bracs_cggnn_3_classes_gin.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        model = CellGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=node_dim,
            num_classes=3
        )

        # 3. run the explainer
        explainer = GraphGradCAMExplainer(
            model=model,
            save_path=self.out_path
        )
        importance_scores, logits = explainer.process(
            graph,
            output_name=self.graph_name.replace('.bin', '')
        )

        # 3. tests
        self.assertIsInstance(importance_scores, np.ndarray)
        self.assertIsInstance(logits, np.ndarray)
        self.assertEqual(graph.number_of_nodes(), importance_scores.shape[0])

    def test_graphgradcam(self):
        """
        Test Graph GradCAM.
        """

        # 1. load a graph
        graph, _ = load_graphs(os.path.join(self.graph_path, self.graph_name))
        graph = graph[0]
        graph = set_graph_on_cuda(graph) if IS_CUDA else graph
        node_dim = graph.ndata['feat'].shape[1]

        # 2. create model
        config_fname = os.path.join(
            self.current_path,
            'config',
            'cg_bracs_cggnn_3_classes_gin.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        model = CellGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=node_dim,
            num_classes=3
        )

        # 3. run the explainer
        explainer = GraphGradCAMExplainer(
            model=model
        )
        importance_scores, logits = explainer.process(graph)

        # 3. tests
        self.assertIsInstance(importance_scores, np.ndarray)
        self.assertIsInstance(logits, np.ndarray)
        self.assertEqual(graph.number_of_nodes(), importance_scores.shape[0])

    def test_graphgradcam_explainer_all(self):
        """
        Test Graph GradCAM by explaining all classes.
        """

        # 1. load a graph
        graph, _ = load_graphs(os.path.join(self.graph_path, self.graph_name))
        graph = graph[0]
        graph = set_graph_on_cuda(graph) if IS_CUDA else graph
        node_dim = graph.ndata['feat'].shape[1]
        class_idx = [0, 1, 2]

        # 2. create model
        config_fname = os.path.join(
            self.current_path,
            'config',
            'cg_bracs_cggnn_3_classes_gin.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        model = CellGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=node_dim,
            num_classes=3
        )

        # 3. run the explainer
        explainer = GraphGradCAMExplainer(
            model=model
        )
        importance_scores, logits = explainer.process(
            graph, class_idx=class_idx)

        # 3. tests
        self.assertIsInstance(importance_scores, np.ndarray)
        self.assertIsInstance(logits, np.ndarray)
        self.assertEqual(len(class_idx), importance_scores.shape[0])
        self.assertEqual(graph.number_of_nodes(), importance_scores.shape[1])

    def test_graphgradcam_explainer_one_class(self):
        """
        Test Graph GradCAM by explaining one specific class.
        """

        # 1. load a graph
        graph, _ = load_graphs(os.path.join(self.graph_path, self.graph_name))
        graph = graph[0]
        graph = set_graph_on_cuda(graph) if IS_CUDA else graph
        node_dim = graph.ndata['feat'].shape[1]
        class_idx = 0

        # 2. create model
        config_fname = os.path.join(
            self.current_path,
            'config',
            'cg_bracs_cggnn_3_classes_gin.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        model = CellGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=node_dim,
            num_classes=3
        )

        # 3. run the explainer
        explainer = GraphGradCAMExplainer(
            model=model
        )
        importance_scores, logits = explainer.process(
            graph, class_idx=class_idx)

        # 3. tests
        self.assertIsInstance(importance_scores, np.ndarray)
        self.assertIsInstance(logits, np.ndarray)
        self.assertEqual(graph.number_of_nodes(), importance_scores.shape[0])

    def test_graphgradcampp(self):
        """
        Test Graph GradCAM++.
        """

        # 1. load a graph
        graph, _ = load_graphs(os.path.join(self.graph_path, self.graph_name))
        graph = graph[0]
        graph = set_graph_on_cuda(graph) if IS_CUDA else graph
        node_dim = graph.ndata['feat'].shape[1]

        # 2. create model
        config_fname = os.path.join(
            self.current_path,
            'config',
            'cg_bracs_cggnn_3_classes_gin.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        model = CellGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=node_dim,
            num_classes=3
        )

        # 2. run the explainer
        explainer = GraphGradCAMPPExplainer(
            model=model
        )
        importance_scores, logits = explainer.process(graph)

        # 3. tests
        self.assertIsInstance(importance_scores, np.ndarray)
        self.assertIsInstance(logits, np.ndarray)
        self.assertEqual(graph.number_of_nodes(), importance_scores.shape[0])

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    unittest.main()
