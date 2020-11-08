from typing import List

import torch
from torch import nn


class GraphBCELoss(nn.Module):
    """Binary Cross Entropy loss over each label seperately, then averaged"""
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the logit and targets

        Args:
            logits (torch.Tensor): Logits for the graph with the shape: B x nr_classes
            targets (torch.Tensor): Targets one-hot encoded with the shape: B x nr_classes

        Returns:
            torch.Tensor: Graph loss
        """
        return self.bce(input=logits, target=targets.to(torch.float32))


class GraphSoftMacroF1Loss(nn.Module):
    """Soft variant of the macro F1 score as a loss function"""
    def __init__(self) -> None:
        super().__init__()
        self.smooth = 1e-16

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the logit and targets

        Args:
            logits (torch.Tensor): Logits for the graph with the shape: B x nr_classes
            targets (torch.Tensor): Targets one-hot encoded with the shape: B x nr_classes

        Returns:
            torch.Tensor: Graph loss
        """
        soft_tp = torch.sum(logits * targets, dim=0)
        soft_fp = torch.sum(logits * (1 - targets), dim=0)
        soft_fn = torch.sum((1 - logits) * targets, dim=0)
        soft_f1 = 2 * soft_tp / (2 * soft_tp + soft_fn + soft_fp + self.smooth)
        cost = 1 - soft_f1
        macro_cost = torch.mean(cost)
        return macro_cost


class NodeStochasticCrossEntropy(nn.Module):
    def __init__(self, drop_probability=0.0, background_label=4) -> None:
        super().__init__()
        assert 0.0 <= drop_probability <= 1.0, f"drop_probability must be valid proability but is {drop_probability}"
        self.drop_probability = drop_probability
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=background_label)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, graph_associations: List[int]
    ) -> torch.Tensor:
        """Compute the loss of the given logits and target labels

        Args:
            logits (torch.Tensor): Logits for the nodes with the shape: \sum_{i=0}^B nr_nodes x nr_classes
            targets (torch.Tensor): Targets labels with the shape: \sum_{i=0}^B nr_nodes
            graph_associations (List[int]): Information needed to unbatch logits and targets

        Returns:
            torch.Tensor: Node loss
        """
        if self.drop_probability > 0:
            to_keep_mask = torch.rand(targets.shape[0]) > self.drop_probability
            targets = targets[to_keep_mask]
            logits = logits[to_keep_mask]
        targets = targets.to(torch.int64)
        return self.cross_entropy(logits, targets)
