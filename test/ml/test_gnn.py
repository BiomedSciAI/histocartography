"""Unit test for ml.layers.gnn"""
import unittest
import torch.nn as nn

from histocartography.ml.layers.gnn import GINLayer


class GNNTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""

    def test_download_file_to_local(self):
        """Test download_file_to_local()."""
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
