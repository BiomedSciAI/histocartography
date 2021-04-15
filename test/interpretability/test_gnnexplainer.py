"""Unit test for interpretability.graph_pruning_explainer"""
import unittest
import numpy as np
import cv2
import torch
import yaml
from copy import deepcopy
import os
import shutil
from dgl.data.utils import load_graphs

from histocartography.interpretability import GraphPruningExplainer
from histocartography.ml import CellGraphModel
from histocartography.utils import set_graph_on_cuda, download_test_data

IS_CUDA = torch.cuda.is_available()


class GraphGNNExplainer(unittest.TestCase):
    """GraphGNNExplainer class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        download_test_data(self.data_path)
        self.graph_path = os.path.join(self.data_path, 'tissue_graphs')
        self.graph_name = '283_dcis_4.bin'
        self.out_path = os.path.join(self.data_path, 'graph_pruning_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)

    def test_gnn_explainer(self):
        """
        Test GNNExplainer (ie, graph_pruning_explainer).
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
        explainer = GraphPruningExplainer(
            model=model,

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
