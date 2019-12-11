"""Unit test for ml.model.diffpool"""
import unittest
import torch
import dgl

from histocartography.ml.models.multi_level_graph_model import MultiLevelGraphModel
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN


class MultiGraphTestCase(unittest.TestCase):
    """MultiGraphTestCase class."""

    def setUp(self):
        """Setting up the test."""

    def test_multi_level_graph_model(self):
        """Test multi level graph model."""

        # 1. generate dummy data for the forward pass.
        batch_size = 16
        ll_num_nodes = 100
        hl_num_nodes = 10

        ll_node_dim = 13
        hl_node_dim = 14

        # Build the low level DGL graphs
        ll_dgl_graphs = [dgl.DGLGraph() for _ in range(batch_size)]
        for g in ll_dgl_graphs:
            g.add_nodes(ll_num_nodes)
            g.add_edges([0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0])
            g.ndata[GNN_NODE_FEAT_IN] = torch.randn(ll_num_nodes, ll_node_dim)
        ll_dgl_graphs = dgl.batch(ll_dgl_graphs)

        # Build the high level graphs
        hl_dgl_graphs = [dgl.DGLGraph() for _ in range(batch_size)]
        for g in hl_dgl_graphs:
            g.add_nodes(hl_num_nodes)
            g.add_edges([0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0])
            g.ndata[GNN_NODE_FEAT_IN] = torch.randn(hl_num_nodes, hl_node_dim)
        hl_dgl_graphs = dgl.batch(hl_dgl_graphs)

        # Build assignment matrix from low level graph to high level graph
        assignment_matrix = [torch.empty(hl_num_nodes, ll_num_nodes).random_(2) for _ in range(batch_size)]

        # 2. create a Diff Pool model.
        config = {
              "model_type": "MultiGraph",
              "model_params": {
                "num_classes": 2,
                "use_bn": False,
                "dropout": 0.0,
                "cat": False,
                "gnn_params": [
                  {
                    "layer_type": "gin_layer",
                    "activation": "relu",
                    "hidden_dim": 11,
                    "output_dim": 32,
                    "n_layers": 2,
                    "neighbor_pooling_type": "mean"
                  },
                  {
                    "layer_type": "gin_layer",
                    "activation": "relu",
                    "hidden_dim": 10,
                    "output_dim": 78,
                    "n_layers": 2,
                    "neighbor_pooling_type": "mean"
                  }
                ]
                }
        }

        model = MultiLevelGraphModel(
            config=config["model_params"],
            ll_node_dim=ll_node_dim,
            hl_node_dim=hl_node_dim
        )

        # 3. print the model parameters
        print_params = True
        if print_params:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name, param.shape)

        # 4. forward pass
        logits = model(ll_dgl_graphs, hl_dgl_graphs, assignment_matrix)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    multi_level_graph = MultiGraphTestCase()
    multi_level_graph.test_multi_level_graph_model()
