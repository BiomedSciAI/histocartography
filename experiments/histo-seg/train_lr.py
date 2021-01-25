import logging
import os
import pickle
from typing import Dict, List, Optional

import dgl
import matplotlib.pyplot as plt
import mlflow
import torch
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder

from dataset import GraphClassificationDataset
from logging_helper import prepare_experiment, robust_mlflow
from losses import get_loss
from models import (
    ImageTissueClassifier,
    SemiSuperPixelTissueClassifier,
    SuperPixelTissueClassifier,
)
from utils import dynamic_import_from, get_config


def get_model(
    gnn_config: dict,
    NR_CLASSES: int,
    graph_classifier_config: Optional[dict] = None,
    node_classifier_config: Optional[dict] = None,
):
    if graph_classifier_config is None:
        assert (
            node_classifier_config is not None
        ), "Either graph classifier or node classifier must be used"
        model = SuperPixelTissueClassifier(
            gnn_config=gnn_config,
            node_classifier_config=node_classifier_config,
            nr_classes=NR_CLASSES,
        )
        mode = "strong_supervision"
    elif node_classifier_config is None:
        model = ImageTissueClassifier(
            gnn_config=gnn_config,
            graph_classifier_config=graph_classifier_config,
            nr_classes=NR_CLASSES,
        )
        mode = "weak_supervision"
    else:
        model = SemiSuperPixelTissueClassifier(
            gnn_config=gnn_config,
            graph_classifier_config=graph_classifier_config,
            node_classifier_config=node_classifier_config,
            nr_classes=NR_CLASSES,
        )
        mode = "semi_strong_supervision"
    return model, mode


def get_logit_information(logits, mode):
    if mode == "strong_supervision":
        return {"node_logits": logits}
    elif mode == "weak_supervision":
        return {"graph_logits": logits}
    elif mode == "semi_strong_supervision":
        graph_logits, node_logits = logits
        return {"node_logits": node_logits, "graph_logits": graph_logits}
    else:
        raise NotImplementedError(f"Logit mode {mode} not implemented")


class CombinedCriterion(torch.nn.Module):
    def __init__(self, loss: dict, mode: str, device) -> None:
        super().__init__()
        if mode in ["strong_supervision", "semi_strong_supervision"]:
            self.node_criterion = get_loss(loss, "node", device)
        if mode in ["weak_supervision", "semi_strong_supervision"]:
            self.graph_criterion = get_loss(loss, "graph", device)
        if mode == "semi_strong_supervision":
            self.node_loss_weight = loss.get("node_weight", 0.5)
            assert (
                0.0 <= self.node_loss_weight <= 1.0
            ), f"Node weight loss must be between 0 and 1, but is {self.node_loss_weight}"
            self.graph_loss_weight = 1.0 - self.node_loss_weight
        self.mode = mode

    def forward(
        self,
        node_logits: Optional[torch.Tensor] = None,
        node_labels: Optional[torch.Tensor] = None,
        graph_logits: Optional[torch.Tensor] = None,
        graph_labels: Optional[torch.Tensor] = None,
        node_associations: Optional[List[int]] = None,
    ):
        if self.mode == "strong_supervision":
            assert (
                node_logits is not None and node_labels is not None
            ), "Cannot use node criterion without input"
            return self.node_criterion(
                logits=node_logits,
                targets=node_labels,
                graph_associations=node_associations,
            )
        elif self.mode == "weak_supervision":
            assert (
                graph_logits is not None and graph_labels is not None
            ), "Cannot use graph criterion without input"
            return self.graph_criterion(logits=graph_logits, targets=graph_labels)
        elif self.mode == "semi_strong_supervision":
            assert (
                node_logits is not None and node_labels is not None
            ), "Cannot use combined criterion without node input"
            assert (
                graph_logits is not None and graph_labels is not None
            ), "Cannot use combined criterion without graph input"
            node_loss = self.node_criterion(
                logits=node_logits,
                targets=node_labels,
                graph_associations=node_associations,
            )
            graph_loss = self.graph_criterion(logits=graph_logits, targets=graph_labels)
            combined_loss = (
                self.graph_loss_weight * graph_loss + self.node_loss_weight * node_loss
            )
            return combined_loss
        else:
            raise NotImplementedError(f"Criterion mode {self.mode} not implemented")


def find_learning_rate(
    dataset: str,
    model_config: Dict,
    data_config: Dict,
    metrics_config: Dict,
    batch_size: int,
    nr_epochs: int,
    num_workers: int,
    optimizer: Dict,
    loss: Dict,
    test: bool,
    validation_frequency: int,
    clip_gradient_norm: Optional[float] = None,
    **kwargs,
) -> None:
    """Train the classification model for a given number of epochs.

    Args:
        model_config (Dict): Configuration of the models (gnn and classifier)
        data_config (Dict): Configuration of the data (e.g. splits)
        batch_size (int): Batch size
        nr_epochs (int): Number of epochs to train
        optimizer (Dict): Configuration of the optimizer
    """
    logging.info(f"Unmatched arguments for training: {kwargs}")

    if test:
        data_config["overfit_test"] = True
        num_workers = 0

    BACKGROUND_CLASS = dynamic_import_from(dataset, "BACKGROUND_CLASS")
    NR_CLASSES = dynamic_import_from(dataset, "NR_CLASSES")
    prepare_graph_datasets = dynamic_import_from(dataset, "prepare_graph_datasets")

    # Data loaders
    training_dataset, validation_dataset = prepare_graph_datasets(**data_config)

    # More monkeypatches
    old_method = GraphClassificationDataset.__getitem__

    def new_method(*args, **kwargs):
        outputs = old_method(*args, **kwargs)
        return outputs[0], outputs[2]

    GraphClassificationDataset.__getitem__ = new_method

    def collate_special(samples):
        graphs, node_labels = map(list, zip(*samples))
        return dgl.batch(graphs), torch.cat(node_labels)

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_special,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        collate_fn=collate_special,
        num_workers=num_workers,
    )

    # Compute device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        robust_mlflow(mlflow.log_param, "device", torch.cuda.get_device_name(0))
    else:
        robust_mlflow(mlflow.log_param, "device", "CPU")

    # Model
    model, mode = get_model(NR_CLASSES=NR_CLASSES, **model_config)
    model = model.to(device)
    nr_trainable_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    robust_mlflow(mlflow.log_param, "nr_parameters", nr_trainable_total_params)
    robust_mlflow(mlflow.log_param, "supervision", mode)

    # Loss function
    criterion = CombinedCriterion(loss=loss, mode=mode, device=device)

    # Optimizer
    optimizer_class = dynamic_import_from("torch.optim", optimizer["class"])
    optim = optimizer_class(model.parameters(), **optimizer["params"])

    lr_finder = LRFinder(model, optim, criterion, device=device)

    # Fast AI Method
    lr_finder.range_test(
        training_loader, start_lr=1e-7, end_lr=1, num_iter=10000, diverge_th=30
    )
    fig, ax = plt.subplots(figsize=(15, 10))
    lr_finder.plot(ax=ax)
    with open("lr_range_test_fast_ai.pickle", "wb") as f:
        pickle.dump(fig, f)
    mlflow.log_artifact("lr_range_test_fast_ai.pickle")
    os.remove("lr_range_test_fast_ai.pickle")
    fig.savefig("lr_range_test_fast_ai.png", dpi=300)
    mlflow.log_artifact("lr_range_test_fast_ai.png")
    os.remove("lr_range_test_fast_ai.png")
    lr_finder.reset()

    # Leslie Smith Method
    lr_finder.range_test(
        training_loader,
        val_loader=validation_loader,
        start_lr=1e-6,
        end_lr=1e-3,
        num_iter=1000,
        step_mode="linear",
        diverge_th=30,
    )
    fig, ax = plt.subplots(figsize=(15, 10))
    lr_finder.plot(ax=ax, log_lr=False)
    with open("lr_range_test_leslie_smith.pickle", "wb") as f:
        pickle.dump(fig, f)
    mlflow.log_artifact("lr_range_test_leslie_smith.pickle")
    os.remove("lr_range_test_leslie_smith.pickle")
    fig.savefig("lr_range_test_leslie_smith.png", dpi=300)
    mlflow.log_artifact("lr_range_test_leslie_smith.png")
    os.remove("lr_range_test_leslie_smith.png")
    lr_finder.reset()


if __name__ == "__main__":
    # Ugly monkeypatch
    def _move_to_device(self, inputs, labels, non_blocking=True):
        def move(obj, device, non_blocking):
            if hasattr(obj, "to"):
                return obj.to(device)
            elif isinstance(obj, tuple):
                return tuple(move(o, device, non_blocking) for o in obj)
            elif isinstance(obj, list):
                return [move(o, device, non_blocking) for o in obj]
            elif isinstance(obj, dict):
                return {k: move(o, device, non_blocking) for k, o in obj.items()}
            else:
                return obj

        inputs = move(inputs, self.device, non_blocking=non_blocking)
        labels = move(labels, self.device, non_blocking=non_blocking)
        return inputs, labels

    LRFinder._move_to_device = _move_to_device

    config, config_path, test = get_config(
        name="train",
        default="default.yml",
        required=("model", "data", "metrics", "params"),
    )
    prepare_experiment(config_path=config_path, **config)
    robust_mlflow(mlflow.log_param, "task", "find_lr")
    find_learning_rate(
        model_config=config["model"],
        data_config=config["data"],
        metrics_config=config["metrics"],
        config_path=config_path,
        test=test,
        **config["params"],
    )
