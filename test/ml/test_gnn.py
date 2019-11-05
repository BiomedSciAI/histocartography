"""Unit test for ml.layers.gnn"""
import unittest
import torch.nn as nn

from histocartography.ml.layers.gnn import GINLayer


class GNNTestCase(unittest.TestCase):
    """GNNTestCase class."""

    def setUp(self):
        """Setting up the test."""

    def test_gnn(self):
        """Test gnn."""
        gnn_model = GINLayer(
            16,
            32,
            16,
            'relu',
            0
        )

        self.assertTrue(isinstance(gnn_model, nn.Module))

    def tearDown(self):
        """Tear down the tests."""
