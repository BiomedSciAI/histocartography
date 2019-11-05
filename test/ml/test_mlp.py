"""Unit test for ml.layers.mlp"""
import unittest
import torch.nn as nn

from histocartography.ml.layers.mlp import MLP


class MLPTestCase(unittest.TestCase):
    """MLPTestCase class."""

    def setUp(self):
        """Setting up the test."""

    def test_mlp(self):
        """Test mlp."""
        in_dim = 16
        h_dim = 32
        out_dim = 10
        num_layers = 4

        mlp = MLP(in_dim, h_dim, out_dim, num_layers)

        self.assertTrue(isinstance(mlp, nn.Module))

    def tearDown(self):
        """Tear down the tests."""
