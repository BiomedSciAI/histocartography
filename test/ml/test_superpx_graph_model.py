"""Unit test for ml.model.diffpool"""
import unittest
import torch
import dgl

from histocartography.ml.models.superpx_graph_model import SuperpxGraphModel
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN


class SuperpxGraphTestCase(unittest.TestCase):
    """MultiGraphTestCase class."""

    def setUp(self):
        """Setting up the test."""

    def test_superpx_graph_model(self):
        """Test cell graph model."""

        # 1. generate dummy data for the forward pass.
        batch_size = 16
        num_nodes = 100
        node_dim = 13

        # Build the superpx DGL graphs
        dgl_graphs = [dgl.DGLGraph() for _ in range(batch_size)]
        for g in dgl_graphs:
            g.add_nodes(num_nodes)
            g.add_edges([0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0])
            g.ndata[GNN_NODE_FEAT_IN] = torch.randn(num_nodes, node_dim)
        dgl_graphs = dgl.batch(dgl_graphs)

        # 2. set dummy config
        config = {
            "graph_building": {
              "graph_building_type": "knn_graph_builder",
              "max_num_nodes": 5000,
              "edge_encoding": False,
              "edge_threshold": 0.5
            },
            "model_params": {
            "gnn_params": {
                "layer_type": "gin_layer",
                "activation": "relu",
                "n_layers": 2,
                "neighbor_pooling_type": "mean",
                "hidden_dim": 11,
                "output_dim": 32
              },
              "num_classes": 3,
              "dropout": 0.0,
              "use_bn": False,
              "cat": True,
              "node_dim": 2,
              "activation": "relu",
              "readout": {
                "num_layers": 2,
                "hidden_dim": 64,
              }
            },
            "model_type": "CellGraphModel"
        }

        model = SuperpxGraphModel(
            config=config["model_params"],
            node_dim=node_dim,
        )

        # 3. print the model parameters
        print_params = True
        if print_params:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name, param.shape)

        # 4. forward pass
        logits = model(dgl_graphs)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    model = SuperpxGraphTestCase()
    model.test_superpx_graph_model()
