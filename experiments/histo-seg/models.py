from typing import Dict, Tuple

import dgl
import torch
import torchvision
from histocartography.ml.layers.multi_layer_gnn import MultiLayerGNN
from torch import nn

from constants import GNN_NODE_FEAT_IN, GNN_NODE_FEAT_OUT
from utils import dynamic_import_from


class ClassifierHead(nn.Module):
    """A basic classifier head"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 2,
        hidden_dim: int = None,
        activation: str = "ReLU",
        input_dropout: float = 0.0,
        layer_dropout: float = 0.0,
    ) -> None:
        """Create a basic classifier head

        Args:
            input_dim (int): Dimensionality of the input
            output_dim (int): Number of output classes
            n_layers (int, optional): Number of layers (including input to hidden and hidden to output layer). Defaults to 2.
            hidden_dim (int, optional): Dimensionality of the hidden layers. Defaults to None.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        activation = dynamic_import_from("torch.nn", activation)
        modules = []
        if input_dropout > 0:
            modules.append(nn.Dropout(input_dropout))
        if n_layers > 1:
            modules.append(nn.Linear(input_dim, hidden_dim))
            modules.append(activation())
        for _ in range(n_layers - 2):
            if layer_dropout > 0:
                modules.append(nn.Dropout(layer_dropout))
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(activation())
        if n_layers == 1:
            modules.append(nn.Linear(input_dim, output_dim))
        else:
            modules.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do a forward pass through the classifier head

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)


class NodeClassifierHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        node_classifier_config: Dict,
        nr_classes: int = 4,
    ) -> None:
        super().__init__()
        self.seperate_heads = node_classifier_config.pop("seperate_heads", True)
        if self.seperate_heads:
            node_classifiers = [
                ClassifierHead(
                    input_dim=latent_dim, output_dim=1, **node_classifier_config
                )
                for _ in range(nr_classes)
            ]
            self.node_classifiers = nn.ModuleList(node_classifiers)
        else:
            self.node_classifier = ClassifierHead(
                input_dim=latent_dim, output_dim=nr_classes, **node_classifier_config
            )

    def forward(self, node_embedding: torch.Tensor) -> torch.Tensor:
        if self.seperate_heads:
            node_logit = torch.empty(
                (node_embedding.shape[0], len(self.node_classifiers)),
                device=node_embedding.device,
            )
            for i, node_classifier in enumerate(self.node_classifiers):
                classifier_output = node_classifier(node_embedding).squeeze(1)
                node_logit[:, i] = classifier_output
            return node_logit
        else:
            return self.node_classifier(node_embedding)


class GraphClassifierHead(nn.Module):
    def __init__(
        self, latent_dim: int, graph_classifier_config: Dict, nr_classes: int = 4
    ) -> None:
        super().__init__()
        self.graph_classifier = ClassifierHead(
            input_dim=latent_dim,
            output_dim=nr_classes,
            **graph_classifier_config,
        )

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.graph_classifier(graph_embedding)


class SuperPixelTissueClassifier(nn.Module):
    def __init__(
        self,
        gnn_config: Dict,
        node_classifier_config: Dict,
        nr_classes: int = 4,
    ) -> None:
        super().__init__()
        self.gnn_model = MultiLayerGNN(gnn_config)
        if gnn_config["agg_operator"] in ["none", "lstm"]:
            latent_dim = gnn_config["output_dim"]
        elif gnn_config["agg_operator"] in ["concat"]:
            latent_dim = gnn_config["output_dim"] * gnn_config["n_layers"]
        else:
            raise NotImplementedError(
                f"Only supported agg operators are [none, lstm, concat]"
            )
        self.node_classifier = NodeClassifierHead(
            latent_dim, node_classifier_config, nr_classes
        )

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        in_features = graph.ndata[GNN_NODE_FEAT_IN]
        self.gnn_model(graph, in_features)
        node_embedding = graph.ndata[GNN_NODE_FEAT_OUT]
        node_logit = self.node_classifier(node_embedding)
        return node_logit


class SemiSuperPixelTissueClassifier(nn.Module):
    """Classifier that uses both weak graph labels and some provided node labels"""

    def __init__(
        self,
        gnn_config: Dict,
        graph_classifier_config: Dict,
        node_classifier_config: Dict,
        nr_classes: int = 4,
    ) -> None:
        """Build a classifier to classify superpixel tissue graphs

        Args:
            config (Dict): Configuration of the models
            nr_classes (int, optional): Number of classes to consider. Defaults to 4.
        """
        super().__init__()
        self.gnn_model = MultiLayerGNN(gnn_config)
        if gnn_config["agg_operator"] in ["none", "lstm"]:
            latent_dim = gnn_config["output_dim"]
        elif gnn_config["agg_operator"] in ["concat"]:
            latent_dim = gnn_config["output_dim"] * gnn_config["n_layers"]
        else:
            raise NotImplementedError(
                f"Only supported agg operators are [none, lstm, concat]"
            )
        self.graph_classifier = GraphClassifierHead(
            latent_dim, graph_classifier_config, nr_classes
        )
        self.node_classifiers = NodeClassifierHead(
            latent_dim, node_classifier_config, nr_classes
        )

    def forward(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass on the graph

        Args:
            graph (dgl.DGLGraph): Input graph with node features in GNN_NODE_FEAT_IN

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits of the graph classifier, logits of the node classifiers
        """
        in_features = graph.ndata[GNN_NODE_FEAT_IN]
        graph_embedding = self.gnn_model(graph, in_features)
        graph_logit = self.graph_classifier(graph_embedding)
        node_embedding = graph.ndata[GNN_NODE_FEAT_OUT]
        node_logit = self.node_classifiers(node_embedding)
        return graph_logit, node_logit


class ImageTissueClassifier(nn.Module):
    """Classifier that uses both weak graph labels and some provided node labels"""

    def __init__(
        self,
        gnn_config: Dict,
        graph_classifier_config: Dict,
        nr_classes: int = 4,
    ) -> None:
        """Build a classifier to classify superpixel tissue graphs

        Args:
            config (Dict): Configuration of the models
            nr_classes (int, optional): Number of classes to consider. Defaults to 4.
        """
        super().__init__()
        self.gnn_model = MultiLayerGNN(gnn_config)
        if gnn_config["agg_operator"] in ["none", "lstm"]:
            latent_dim = gnn_config["output_dim"]
        elif gnn_config["agg_operator"] in ["concat"]:
            latent_dim = gnn_config["output_dim"] * gnn_config["n_layers"]
        else:
            raise NotImplementedError(
                f"Only supported agg operators are [none, lstm, concat]"
            )
        self.graph_classifier = GraphClassifierHead(
            latent_dim, graph_classifier_config, nr_classes
        )

    def forward(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass on the graph

        Args:
            graph (dgl.DGLGraph): Input graph with node features in GNN_NODE_FEAT_IN

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits of the graph classifier, logits of the node classifiers
        """
        in_features = graph.ndata[GNN_NODE_FEAT_IN]
        graph_embedding = self.gnn_model(graph, in_features)
        graph_logit = self.graph_classifier(graph_embedding)
        return graph_logit


class PatchTissueClassifier(nn.Module):
    def __init__(self, freeze: int = 0, **kwargs) -> None:
        super().__init__()
        self.model = self._select_model(freeze=freeze, **kwargs)
        self.freeze = freeze

    def _select_model(
        self,
        architecture: str,
        num_classes: int,
        dropout: int,
        freeze: int = 0,
        **kwargs,
    ) -> Tuple[nn.Module, int]:
        """Returns the model and number of features for a given name

        Args:
            architecture (str): Name of architecture. Can be [resnet{18,34,50,101,152}, vgg{16,19}]

        Returns:
            Tuple[nn.Module, int]: The model and the number of features
        """
        model_class = dynamic_import_from("torchvision.models", architecture)
        model = model_class(**kwargs)
        if isinstance(model, torchvision.models.resnet.ResNet):
            feature_dim = model.fc.in_features
            model.fc = nn.Linear(feature_dim, num_classes)
        else:
            feature_dim = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout), nn.Linear(feature_dim, num_classes)
            )
        self._freeze_layers(model, freeze)
        return model

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _freeze_layers(model, freeze):
        if isinstance(model, torchvision.models.resnet.ResNet):
            for layer in list(model.children())[:freeze]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            for param in model.features[:freeze].parameters():
                param.requires_grad = False

    def freeze_encoder(self):
        if isinstance(self.model, torchvision.models.resnet.ResNet):
            for layer in list(self.model.children())[:-1]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            for param in self.model.features.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        if isinstance(self.model, torchvision.models.resnet.ResNet):
            for layer in list(self.model.children())[:-1]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            for param in self.model.features.parameters():
                param.requires_grad = True
        self._freeze_layers(self.model, self.freeze)


class SegmentationFromCNN(nn.Module):
    def __init__(self, base_model, upsample_mode="bicubic"):
        super().__init__()
        self.base_model = base_model.model
        last_layer_weights = self.base_model.classifier[-1].weight.data
        last_layer_bias = self.base_model.classifier[-1].bias.data
        self.base_model.classifier = self.base_model.classifier[:-1]
        self.post_process = nn.Sequential(
            nn.AvgPool2d(7, 1, padding=3),
            nn.Conv2d(1280, 4, 1),
            nn.Upsample(scale_factor=32, mode="nearest"),
        )
        self.post_process[1].weight.data = last_layer_weights.unsqueeze(2).unsqueeze(3)
        self.post_process[1].bias.data = last_layer_bias

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.base_model.classifier(x)
        x = self.post_process(x)
        return x


class MLPModel(nn.Module):
    def __init__(self, num_classes=4, mode='multihead'):
        """
        Model constructor
        Args:
            num_classes (int): Number of classes. Typically 4 if multi-label
                                  learning or multi-head learning, 6 if final
                                  gleason grade prediction. 
            mode (str): Classification mode. 3 options:
                            - multihead: as in SICAPv2 paper (P+S) classification, then sum 
                            - multilabel: as in the ETH paper, ie independent binary classification.
                            - finalgleasonscore: directly predicting the final gleason grade (ie, 0, 6, 7, 8, 9, 10). 
        """
        super(MLPModel, self).__init__()
        self.backbone = nn.Sequential(
          nn.Linear(4, 8),
          nn.ReLU(),
          nn.Linear(8, 16),
          nn.ReLU()
        )
        self.mode = mode

        if self.mode == 'multihead':
            self.primary_classifier = nn.Linear(16, num_classes)
            self.secondary_classifier = nn.Linear(16, num_classes)
        elif self.mode == 'multilabel' or self.mode == 'finalgleasonscore':
            self.classifier = nn.Linear(16, num_classes)
        else:
            raise ValueError('Unsupported mode')

    def forward(self, x):
        """
        Forward-pass
        :param x: (torch.FloatTensor)
        """
        x = self.backbone(x)
        if self.mode == 'multihead':
            primary = self.primary_classifier(x)
            secondary = self.secondary_classifier(x)
            return primary, secondary
        else:
            logits = self.classifier(x)
            return logits
