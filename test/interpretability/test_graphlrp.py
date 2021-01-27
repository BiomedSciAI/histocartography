"""Unit test for interpretability.lrp_explainer.lrp_gnn_explainer"""
import unittest
import numpy as np
import cv2 
import torch 
import yaml
import os 
from copy import deepcopy
from dgl.data.utils import load_graphs

from histocartography.interpretability.lrp_explainer.lrp_gnn_explainer import LRPGNNExplainer
from histocartography.utils.graph import set_graph_on_cuda

BASE_S3 = 's3://mlflow/'
IS_CUDA = torch.cuda.is_available()


class GraphLRPTestCase(unittest.TestCase):
    """GraphLRPTestCase class."""

    def setUp(self):
        """Setting up the test."""
    
    def test_graphlrp(self):
        """Test Graph LRP.
        """


        base_path = '../data'
        cg_fnames = ['283_dcis_4_cg.bin', '1238_adh_10_cg.bin', '1286_udh_35_cg.bin', '1937_benign_4_cg.bin', '311_fea_25_cg.bin']

        for cg_name in cg_fnames:
            print('*** Testing cell graph explainer GraphGradCAM++ {}'.format(cg_name))

            # 1. load a cell graph
            cell_graph, label_dict = load_graphs(os.path.join(base_path, 'cell_graphs', cg_name))
            cell_graph = set_graph_on_cuda(cell_graph[0]) if IS_CUDA else cell_graph[0]

            # 2. explainer the graph 
            explainer = LRPGNNExplainer(
                model_path=BASE_S3 + '29b7f5ee991e4a3e8b553b49a1c3c05a/artifacts/model_best_val_weighted_f1_score_0'
            )
            importance_score, logits = explainer.process(cell_graph)

            # 3. print output 
            print('Number of nodes:', cell_graph.number_of_nodes())
            print('Number of edges:', cell_graph.number_of_edges())
            print('Node features:', cell_graph.ndata['feat'].shape)
            print('Node centroids:', cell_graph.ndata['centroid'].shape)
            print('Importance scores:', importance_score.shape)
            print('Logits:', logits.shape)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    model = GraphLRPTestCase()
    model.test_graphlrp()
