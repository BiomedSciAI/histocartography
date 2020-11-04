import logging
from typing import Dict, Tuple

import dgl
import torch
from histocartography.ml.layers.multi_layer_gnn import MultiLayerGNN
from torch import nn

from constants import GNN_NODE_FEAT_IN, GNN_NODE_FEAT_OUT


class WeakTissueClassifier(nn.Module):
    """Classifier that uses both weak graph labels and some provided node labels"""
    def __init__(self, config: Dict, nr_classes : int = 4) -> None:
        """Build a classifier to classify superpixel tissue graphs

        Args:
            config (Dict): Configuration of the models
            nr_classes (int, optional): Number of classes to consider. Defaults to 4.
        """
        super().__init__()
        self.gnn_model = MultiLayerGNN(config["gnn"])
        self.latent_dim = config["gnn"]["output_dim"]
        self.graph_classifier = nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_dim, nr_classes),
        )
        self.node_classifiers = [
            nn.Sequential(
                torch.nn.Linear(self.latent_dim, self.latent_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.latent_dim // 2, 1),
            )
            for _ in range(nr_classes)
        ]
        self.nr_classes = nr_classes

    def forward(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass on the graph

        Args:
            graph (dgl.DGLGraph): Input graph with node features in GNN_NODE_FEAT_IN

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits of the graph classifier, logits of the node classifiers
        """
        in_features = graph.ndata[GNN_NODE_FEAT_IN]

        graph_embedding = self.gnn_model(graph, in_features)
        logging.debug(f"Graph embeddings: {graph_embedding.shape}")
        graph_logit = self.graph_classifier(graph_embedding)

        logging.debug(f"In features: {in_features.shape}")
        node_embedding = graph.ndata[GNN_NODE_FEAT_OUT]
        logging.debug(f"Node embeddings: {node_embedding.shape}")
        node_logit = torch.empty((in_features.shape[0], self.nr_classes))
        logging.debug(f"Node logits: {node_logit.shape}")
        for i, node_classifier in enumerate(self.node_classifiers):
            classifier_output = node_classifier(node_embedding).squeeze(1)
            logging.debug(f"Classifier output: {classifier_output.shape}")
            node_logit[:, i] = classifier_output

        return graph_logit, node_logit
