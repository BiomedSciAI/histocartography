"""Unit test for interpretability.pruning_explainer.graph_pruning_explainer"""
import unittest
import numpy as np
import cv2 
import torch 
import yaml
from copy import deepcopy
from dgl.data.utils import load_graphs

from histocartography.interpretability.pruning_explainer.graph_pruning_explainer import GraphPruningExplainer

BASE_S3 = 's3://mlflow/'


class GraphGNNExplainer(unittest.TestCase):
    """GraphGNNExplainer class."""

    def setUp(self):
        """Setting up the test."""
    
    def test_gnnexplainer(self):
        """Test GNN Explainer.
        """

        # 1. load a cell graph
        cell_graph, label_dict = load_graphs('../data/283_dcis_4.bin')
        cell_graph = cell_graph[0]

        # 2. run the explainer  
        explainer = GraphPruningExplainer(
            model_path=BASE_S3 + '6e00ba6464e74150a3dd94a6c2529ad3/artifacts/model_cg_dense_gin'
        )
        importance_score, logits = explainer.process(cell_graph)

        # 3. print graph properties
        print('Number of nodes:', cell_graph.number_of_nodes())
        print('Number of edges:', cell_graph.number_of_edges())
        print('Node features:', cell_graph.ndata['feat'].shape)
        print('Node centroids:', cell_graph.ndata['centroid'].shape)
        print('Importance scores:', importance_score.shape)
        print('Logits:', logits.shape)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    model = GraphGNNExplainer()
    model.test_gnnexplainer()
