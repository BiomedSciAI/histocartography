"""Unit test for ml.models.cell_graph_model"""
import unittest
import torch
import dgl
import os
import yaml
from dgl.data.utils import load_graphs

from histocartography.ml import CellGraphModel
from histocartography.utils.graph import set_graph_on_cuda
from histocartography.utils.io import download_box_link

IS_CUDA = torch.cuda.is_available()


class CGModelTestCase(unittest.TestCase):
    """CGModelTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        self.graph_path = os.path.join(self.data_path, 'cell_graphs')
        self.checkpoint_path = os.path.join(self.data_path, 'checkpoints')
        self.graph_name = '283_dcis_4.bin'
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def test_cell_graph_model(self):
        """Test cell graph model."""

        # 1. Load a cell graph 
        graph, _ = load_graphs(os.path.join(self.graph_path, self.graph_name))
        graph = graph[0]
        graph = set_graph_on_cuda(graph) if IS_CUDA else graph
        node_dim = graph.ndata['feat'].shape[1]

        # 2. load config 
        config_fname = os.path.join(self.current_path, 'config', 'cg_model.yml')
        with open(config_fname, 'r') as file:
            config = yaml.load(file)

        model = CellGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=node_dim,
            num_classes=3
        )

        # 4. forward pass
        logits = model(graph)

        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], 3) 

    def test_pretrained_cell_graph_model(self):
        """Test cell graph model."""

        # 1. load a cell graph 
        graph, _ = load_graphs(os.path.join(self.graph_path, self.graph_name))
        graph = graph[0]
        graph = set_graph_on_cuda(graph) if IS_CUDA else graph

        # 2. load from box and dump
        checkpoint_fname = os.path.join(self.checkpoint_path, 'bracs_cggnn_3_classes_gin.pt')
        download_box_link(
            url='https://ibm.box.com/shared/static/pozkx0ngqjxdr34v5tpmckthts8m12u3.pt',
            out_fname=checkpoint_fname
        )
        model = torch.load(checkpoint_fname)

        # 3. forward pass
        logits = model(graph)

        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], 3) 

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    unittest.main()
