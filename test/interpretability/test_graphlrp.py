"""Unit test for interpretability.lrp_gnn_explainer"""
import unittest
import numpy as np
import cv2 
import torch 
import yaml
import os 
from copy import deepcopy
import shutil
from dgl.data.utils import load_graphs

from histocartography.interpretability import GraphLRPExplainer
from histocartography.utils.graph import set_graph_on_cuda

BASE_S3 = 's3://mlflow/'
IS_CUDA = torch.cuda.is_available()


class GraphLRPTestCase(unittest.TestCase):
    """GraphLRPTestCase class."""

    @classmethod
    def setUpClass(self):
        self.data_path = os.path.join('..', 'data')
        self.graph_path = os.path.join(self.data_path, 'tissue_graphs')
        self.graph_name = '283_dcis_4_tg.bin'
        self.out_path = os.path.join(self.data_path, 'graph_lrp_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path) 
        os.makedirs(self.out_path)

    def test_graphlrp(self):
        """
        Test Graph LRP.
        """

        # 1. load a graph
        graph, _ = load_graphs(os.path.join(self.graph_path, self.graph_name))
        graph = graph[0]
        graph.ndata['feat'] = torch.cat(
            (graph.ndata['feat'][:, :512].float(),  # @TODO: HACK-->truncate features to match pre-trained model. 
            (graph.ndata['centroid']).float()),
            dim=1
        )
        graph = set_graph_on_cuda(graph) if IS_CUDA else graph

        # 2. run the explainer
        explainer = GraphLRPExplainer(
            model_path='https://ibm.box.com/shared/static/t781n2z2w0b0rkbar2opvt38hwslboeh.pt'
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
