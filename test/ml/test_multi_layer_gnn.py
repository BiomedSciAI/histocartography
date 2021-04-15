"""Unit test for ml.layers.multi_layer_gnn"""
import unittest
import torch
import dgl
import yaml
import os

from histocartography.ml import MultiLayerGNN


class MultiLayerGNNTestCase(unittest.TestCase):
    """MultiLayerGNN class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)

    def test_multi_layer_gin(self):
        """
        Test MultiLayerGNN with GIN layers.
        """

        # 1. load dummy config
        config_fname = os.path.join(
            self.current_path,
            'config',
            'multi_layer_gin.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)['model']

        # 2. dummy data
        graph = dgl.rand_graph(100, 10)
        features = torch.rand(100, 512)

        # 2. multi layer GNN
        model = MultiLayerGNN(input_dim=512, **config)
        out = model(graph, features, with_readout=False)

        # 3. tests
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape[0], 100)
        self.assertEqual(out.shape[1], 96)  # 3 layers x 32 hidden dimension

    def test_multi_layer_dense_gin(self):
        """
        Test MultiLayerGNN with dense GIN layers.
        """

        # 1. load dummy config
        config_fname = os.path.join(
            self.current_path,
            'config',
            'multi_layer_dense_gin.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)['model']

        # 2. dummy data
        adjacency = torch.randint(2, (100, 100))
        features = torch.rand(100, 512)

        # 2. multi layer GNN
        model = MultiLayerGNN(input_dim=512, **config)
        out = model(adjacency, features, with_readout=False)

        # 3. tests
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape[0], 100)
        self.assertEqual(out.shape[1], 96)  # 3 layers x 32 hidden dimension

    def test_multi_layer_pna(self):
        """
        Test MultiLayerGNN with PNA layers.
        """

        # 1. load dummy config
        config_fname = os.path.join(
            self.current_path,
            'config',
            'multi_layer_pna.yml')
        with open(config_fname, 'r') as file:
            config = yaml.safe_load(file)['model']

        # 2. dummy data
        graph = dgl.rand_graph(100, 10)
        features = torch.rand(100, 512)

        # 2. multi layer GNN
        model = MultiLayerGNN(input_dim=512, **config)
        out = model(graph, features, with_readout=False)

        # 3. tests
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape[0], 100)
        # 3 agg x 3 scalers x 32 hidden dimension
        self.assertEqual(out.shape[1], 192)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    unittest.main()
