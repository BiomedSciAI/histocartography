"""Unit test for ml.model.diffpool"""
import unittest
import torch
import dgl

from histocartography.ml.models.diff_pool_model import DiffPool
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN


class DiffPoolTestCase(unittest.TestCase):
    """DiffPoolTestCase class."""

    def setUp(self):
        """Setting up the test."""

    def test_diff_pool(self):
        """Test diff pool GNN."""

        # 1. generate dummy data for the forward pass.
        batch_size = 16
        max_num_node = 100
        node_dim = 10
        dgl_graphs = [dgl.DGLGraph() for _ in range(batch_size)]
        for g in dgl_graphs:
            g.add_nodes(max_num_node)
            g.add_edges([0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0])
            g.ndata[GNN_NODE_FEAT_IN] = torch.randn(max_num_node, node_dim)
        dgl_graphs = dgl.batch(dgl_graphs)

        # 2. create a Diff Pool model. @TODO reorganize all this huge mess... hard code what we can.
        config = {
            "model_type": "DiffPool",
            "model_params": {
                "input_dim": 10,
                "hidden_dim": 10,
                "embedding_dim": 10,
                "label_dim": 1,
                "activation": "relu",
                "n_layers": 3,
                "gc_per_block": 3,
                "dropout": 0.0,
                "use_bn": True,
                "n_pooling": 1,
                "neighbor_pooling_type": "mean",
                "pool_ratio": 0.15,
                "cat": False,
                "gnn_before": {
                  "layer_type": "gin_layer",
                  "input_dim": 10,
                  "hidden_dim": 10,
                  "embedding_dim": 10,
                  "label_dim": 1,
                  "activation": "relu",
                  "n_layers": 3,
                  "gc_per_block": 3,
                  "dropout": 0.0,
                  "use_bn": True,
                  "n_pooling":  1,
                  "neighbor_pooling_type": "mean",
                  "pool_ratio": 0.15,
                  "cat": False
                },
                "gnn_after": {
                  "layer_type": "dense_gin_layer",
                  "input_dim": 10,
                  "hidden_dim": 10,
                  "embedding_dim": 10,
                  "label_dim": 1,
                  "activation": "relu",
                  "n_layers": 3,
                  "gc_per_block": 3,
                  "dropout": 0.0,
                  "use_bn": True,
                  "n_pooling":  1,
                  "neighbor_pooling_type": "mean",
                  "pool_ratio": 0.15,
                  "cat": False
                }
              }
        }

        model = DiffPool(config=config["model_params"], max_num_node=max_num_node, batch_size=batch_size)

        # 3. print the model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

        # 4. forward pass
        logits = model(dgl_graphs)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    diff_pool = DiffPoolTestCase()
    diff_pool.test_diff_pool()
