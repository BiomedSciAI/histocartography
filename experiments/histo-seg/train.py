import argparse
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import dgl
import mlflow
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from eth import BACKGROUND_CLASS, NR_CLASSES, prepare_datasets
from logging_helper import GraphClassificationLoggingHelper, log_parameters
from losses import GraphLabelLoss, NodeLabelLoss
from models import WeakTissueClassifier
from utils import dynamic_import_from, start_logging


def collate(
    samples: List[Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]]
) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
    """Aggregate a batch by performing the following:
       Create a graph with disconnected components using dgl.batch
       Stack the graph labels one-hot encoded labels (to shape B x nr_classes)
       Concatenate the node labels to a single vector (graph association can be read from graph.batch_num_nodes)

    Args:
        samples (List[Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]]): List of unaggregated samples

    Returns:
        Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]: Aggregated graph and labels
    """
    graphs, graph_labels, node_labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.stack(graph_labels), torch.cat(node_labels)


def train_graph_classifier(
    model_config: Dict,
    data_config: Dict,
    metrics_config: Dict,
    batch_size: int,
    nr_epochs: int,
    num_workers: int,
    optimizer: Dict,
    node_loss_weight: float,
    node_loss_drop_probability: float,
    config_path: str,
    test: bool
) -> None:
    """Train the classification model for a given number of epochs.

    Args:
        model_config (Dict): Configuration of the models (gnn and classifier)
        data_config (Dict): Configuration of the data (e.g. splits)
        batch_size (int): Batch size
        nr_epochs (int): Number of epochs to train
        optimizer (Dict): Configuration of the optimizer
    """
    assert (
        0.0 <= node_loss_weight <= 1.0
    ), f"Node weight loss must be between 0 and 1, but is {node_loss_weight}"

    if test:
        data_config["overfit_test"] = True

    # MLflow
    mlflow.set_experiment("anv_wsss_train_classifier")
    mlflow.log_artifact(config_path, "config")
    log_parameters(
        data=data_config,
        model=model_config,
        batch_size=batch_size,
        epochs=nr_epochs,
        optimizer=optimizer,
    )
    training_metric_logger = GraphClassificationLoggingHelper(
        metrics_config,
        "train.",
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
    )
    validation_metric_logger = GraphClassificationLoggingHelper(
        metrics_config,
        "valid.",
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
    )

    # Data loaders
    training_dataset, validation_dataset = prepare_datasets(**data_config)
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
    )

    # Compute device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        mlflow.log_param("device", torch.cuda.get_device_name(0))
    else:
        mlflow.log_param("device", "CPU")

    # Model
    model = WeakTissueClassifier(**model_config)
    model = model.to(device)
    nr_trainable_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    mlflow.log_param("nr_parameters", nr_trainable_total_params)

    # Loss function
    graph_loss_weight = 1.0 - node_loss_weight
    graph_criterion = GraphLabelLoss()
    graph_criterion = graph_criterion.to(device)
    node_criterion = NodeLabelLoss(
        background_label=training_dataset.background_index,
        drop_probability=node_loss_drop_probability,
    )
    node_criterion = node_criterion.to(device)

    # Optimizer
    optimizer_class = dynamic_import_from("torch.optim", optimizer["class"])
    optimizer = optimizer_class(model.parameters(), **optimizer["params"])

    for epoch in range(nr_epochs):

        # Train model
        time_before_training = datetime.datetime.now()
        model.train()
        progress_bar = tqdm(
            enumerate(training_loader),
            desc=f"Train Epoch {epoch}",
            total=len(training_loader),
        )
        for iteration, (graph, graph_labels, node_labels) in progress_bar:
            graph = graph.to(device)
            graph_labels = graph_labels.to(device)
            node_labels = node_labels.to(device)

            graph_logits, node_logits = model(graph)

            graph_loss = graph_criterion(graph_logits, graph_labels)
            node_loss = node_criterion(node_logits, node_labels, graph.batch_num_nodes)
            combined_loss = (
                graph_loss_weight * graph_loss + node_loss_weight * node_loss
            )
            combined_loss.backward()
            optimizer.step()

            training_metric_logger.add_iteration_outputs(
                graph_loss=graph_loss.item(),
                node_loss=node_loss.item(),
                graph_logits=graph_logits.detach().cpu(),
                node_logits=node_logits.detach().cpu(),
                graph_labels=graph_labels.cpu(),
                node_labels=node_labels.cpu(),
                node_associations=graph.batch_num_nodes,
            )
        training_metric_logger.log_and_clear(step=epoch)
        training_epoch_duration = (
            datetime.datetime.now() - time_before_training
        ).total_seconds()
        mlflow.log_metric("train.seconds_per_epoch", training_epoch_duration)

        # Validate model
        time_before_validation = datetime.datetime.now()
        model.eval()
        progress_bar = tqdm(
            enumerate(validation_loader),
            desc=f"Valid Epoch {epoch}",
            total=len(validation_loader),
        )
        with torch.no_grad():
            for iteration, (graph, graph_labels, node_labels) in progress_bar:
                graph = graph.to(device)
                graph_labels = graph_labels.to(device)
                node_labels = node_labels.to(device)

                graph_logits, node_logits = model(graph)
                graph_loss = graph_criterion(graph_logits, graph_labels)
                node_loss = node_criterion(
                    node_logits, node_labels, graph.batch_num_nodes
                )

                validation_metric_logger.add_iteration_outputs(
                    graph_loss=graph_loss.item(),
                    node_loss=node_loss.item(),
                    graph_logits=graph_logits.detach().cpu(),
                    node_logits=node_logits.detach().cpu(),
                    graph_labels=graph_labels.cpu(),
                    node_labels=node_labels.cpu(),
                    node_associations=graph.batch_num_nodes,
                )
        validation_metric_logger.log_and_clear(step=epoch, model=model)
        validation_epoch_duration = (
            datetime.datetime.now() - time_before_validation
        ).total_seconds()
        mlflow.log_metric("valid.seconds_per_epoch", validation_epoch_duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default.yml")
    parser.add_argument("--level", type=str, default="WARNING")
    parser.add_argument("--test", action="store_const", const=True, default=False)
    args = parser.parse_args()

    start_logging(args.level)
    assert Path(args.config).exists(), f"Config path does not exist: {args.config}"
    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    assert "train" in config, f"Config does not have an entry train ({config.keys()})"
    config = config["train"]

    logging.info("Start training")
    assert (
        "model" in config
    ), f"model not defined in config {args.config}: {config.keys()}"
    assert (
        "data" in config
    ), f"data not defined in config {args.config}: {config.keys()}"
    assert (
        "metrics" in config
    ), f"metrics not defined in config {args.config}: {config.keys()}"
    assert (
        "params" in config
    ), f"params not defined in config {args.config}: {config.keys()}"
    train_graph_classifier(
        model_config=config["model"],
        data_config=config["data"],
        metrics_config=config["metrics"],
        config_path=args.config,
        test=args.test,
        **config["params"],
    )
