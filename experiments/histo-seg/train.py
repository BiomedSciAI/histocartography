import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import dgl
import mlflow
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from eth import prepare_datasets
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
    optimizer: Dict,
) -> None:
    """Train the classification model for a given number of epochs.

    Args:
        model_config (Dict): Configuration of the models (gnn and classifier)
        data_config (Dict): Configuration of the data (e.g. splits)
        batch_size (int): Batch size
        nr_epochs (int): Number of epochs to train
        optimizer (Dict): Configuration of the optimizer
    """
    mlflow.set_experiment("anv_wsss_train_classifier")
    log_parameters(
        data=data_config,
        model=model_config,
        batch_size=batch_size,
        epochs=nr_epochs,
        optimizer=optimizer,
    )

    # Data loaders
    training_dataset, validation_dataset = prepare_datasets(**data_config)
    training_metric_logger = GraphClassificationLoggingHelper(metrics_config, "train.")
    validation_metric_logger = GraphClassificationLoggingHelper(
        metrics_config, "valid."
    )
    training_loader = DataLoader(
        training_dataset, batch_size=batch_size, collate_fn=collate
    )
    validation_loader = DataLoader(validation_dataset, batch_size=1, collate_fn=collate)

    # Model
    model = WeakTissueClassifier(**model_config)
    nr_trainable_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    mlflow.log_param("nr_parameters", nr_trainable_total_params)

    # Loss function
    graph_criterion = GraphLabelLoss()
    node_criterion = NodeLabelLoss(training_dataset.background_index)

    # Optimizer
    optimizer_class = dynamic_import_from("torch.optim", optimizer["class"])
    optimizer = optimizer_class(model.parameters(), **optimizer["params"])

    for epoch in range(nr_epochs):

        # Train model
        model.train()
        progress_bar = tqdm(
            enumerate(training_loader),
            desc=f"Train Epoch {epoch}",
            total=len(training_loader),
        )
        for iteration, (graph, graph_labels, node_labels) in progress_bar:
            graph_logits, node_logits = model(graph)

            graph_loss = graph_criterion(graph_logits, graph_labels)
            node_loss = node_criterion(node_logits, node_labels, graph.batch_num_nodes)
            combined_loss = graph_loss + node_loss

            combined_loss.backward()
            optimizer.step()

            training_metric_logger.add_iteration_outputs(
                graph_loss=graph_loss.item(),
                node_loss=node_loss.item(),
                graph_logits=graph_logits.detach(),
                node_logits=node_logits.detach(),
                graph_labels=graph_labels,
                node_labels=node_labels,
            )
        training_metric_logger.log_and_clear(step=epoch)

        # Validate model
        model.eval()
        progress_bar = tqdm(
            enumerate(validation_loader),
            desc=f"Valid Epoch {epoch}",
            total=len(validation_loader),
        )
        with torch.no_grad():
            for iteration, (graph, graph_labels, node_labels) in progress_bar:
                graph_logits, node_logits = model(graph)
                graph_loss = graph_criterion(graph_logits, graph_labels)
                node_loss = node_criterion(
                    node_logits, node_labels, graph.batch_num_nodes
                )
                combined_loss = graph_loss + node_loss

                validation_metric_logger.add_iteration_outputs(
                    graph_loss=graph_loss.item(),
                    node_loss=node_loss.item(),
                    graph_logits=graph_logits.detach(),
                    node_logits=node_logits.detach(),
                    graph_labels=graph_labels,
                    node_labels=node_labels,
                )
        validation_metric_logger.log_and_clear(step=epoch, model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default.yml")
    parser.add_argument("--level", type=str, default="WARNING")
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
        **config["params"],
    )
