"""Unit test for ml.layers.mlp"""
import unittest
import torch.nn as nn

from histocartography.ml.layers.mlp import MLP


class MLPTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""

    def test_download_file_to_local(self):
        """Test download_file_to_local()."""
        in_dim = 16
        h_dim = 32
        out_dim = 10
        num_layers = 4

        mlp = MLP(in_dim, h_dim, out_dim, num_layers)

        self.assertTrue(isinstance(mlp, nn.Module))

    def tearDown(self):
        """Tear down the tests."""
