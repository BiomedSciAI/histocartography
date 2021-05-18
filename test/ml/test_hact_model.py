"""Unit test for ml.models.hact_model"""
import unittest
import torch
import dgl
import os
import yaml
from dgl.data.utils import load_graphs

from histocartography.ml import HACTModel
from histocartography.utils import set_graph_on_cuda, download_box_link, download_test_data


IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'


class HACTModelTestCase(unittest.TestCase):
    """HACTModelTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        download_test_data(self.data_path)
        self.model_fname = os.path.join(
            self.data_path, 'models', 'tg_model.pt')
        self.tg_graph_path = os.path.join(self.data_path, 'tissue_graphs')
        self.tg_graph_name = '283_dcis_4.bin'
        self.cg_graph_path = os.path.join(self.data_path, 'cell_graphs')
        self.cg_graph_name = '283_dcis_4.bin'
        self.checkpoint_path = os.path.join(self.data_path, 'checkpoints')
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def test_hact_model(self):
        """Test HACT model."""

        # 1. Load a cell graph
        cell_graph, _ = load_graphs(os.path.join(
            self.cg_graph_path, self.cg_graph_name))
        cell_graph = cell_graph[0]
        cell_graph = set_graph_on_cuda(cell_graph) if IS_CUDA else cell_graph
        cg_node_dim = cell_graph.ndata['feat'].shape[1]

        tissue_graph, _ = load_graphs(os.path.join(
            self.tg_graph_path, self.tg_graph_name))
        tissue_graph = tissue_graph[0]
        tissue_graph = set_graph_on_cuda(
            tissue_graph) if IS_CUDA else tissue_graph
        tg_node_dim = tissue_graph.ndata['feat'].shape[1]

        assignment_matrix = torch.randint(
            2, (tissue_graph.number_of_nodes(), cell_graph.number_of_nodes())).float()
        assignment_matrix = assignment_matrix.cuda() if IS_CUDA else assignment_matrix
        assignment_matrix = [assignment_matrix]  # ie. batch size is 1.

        # 2. load config
        config_fname = os.path.join(
            self.current_path, 'config', 'hact_model.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        model = HACTModel(
            cg_gnn_params=config['cg_gnn_params'],
            tg_gnn_params=config['tg_gnn_params'],
            classification_params=config['classification_params'],
            cg_node_dim=cg_node_dim,
            tg_node_dim=tg_node_dim,
            num_classes=3
        ).to(DEVICE)

        # 4. forward pass
        logits = model(cell_graph, tissue_graph, assignment_matrix)

        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], 3)

    def test_hact_model_bracs_hact_7_classes_pna(self):
        """Test HACT bracs_hact_7_classes_pna model."""

        # 1. Load a cell graph
        cell_graph, _ = load_graphs(os.path.join(
            self.cg_graph_path, self.cg_graph_name))
        cell_graph = cell_graph[0]
        cell_graph = set_graph_on_cuda(cell_graph) if IS_CUDA else cell_graph
        cg_node_dim = cell_graph.ndata['feat'].shape[1]

        tissue_graph, _ = load_graphs(os.path.join(
            self.tg_graph_path, self.tg_graph_name))
        tissue_graph = tissue_graph[0]
        tissue_graph = set_graph_on_cuda(
            tissue_graph) if IS_CUDA else tissue_graph
        tg_node_dim = tissue_graph.ndata['feat'].shape[1]

        assignment_matrix = torch.randint(
            2, (tissue_graph.number_of_nodes(), cell_graph.number_of_nodes())).float()
        assignment_matrix = assignment_matrix.cuda() if IS_CUDA else assignment_matrix
        assignment_matrix = [assignment_matrix]  # ie. batch size is 1.

        # 2. load config and build model with pretrained weights
        config_fname = os.path.join(
            self.current_path,
            'config',
            'bracs_hact_7_classes_pna.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)

        model = HACTModel(
            cg_gnn_params=config['cg_gnn_params'],
            tg_gnn_params=config['tg_gnn_params'],
            classification_params=config['classification_params'],
            cg_node_dim=cg_node_dim,
            tg_node_dim=tg_node_dim,
            num_classes=7,
            pretrained=True
        ).to(DEVICE)

        # 3. forward pass
        logits = model(cell_graph, tissue_graph, assignment_matrix)

        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], 7)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    unittest.main()
