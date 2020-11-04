import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import dgl
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from eth import prepare_datasets
from losses import GraphLabelLoss, NodeLabelLoss
from models import WeakTissueClassifier
from utils import dynamic_import_from, merge_metadata, start_logging


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
    training_dataset, validation_dataset = prepare_datasets(**data_config)
    training_loader = DataLoader(
        training_dataset, batch_size=batch_size, collate_fn=collate
    )
    validation_loader = DataLoader(validation_dataset, batch_size=1, collate_fn=collate)

    model = WeakTissueClassifier(model_config)
    graph_criterion = GraphLabelLoss()
    node_criterion = NodeLabelLoss(training_dataset.background_index)

    optimizer_class = dynamic_import_from("torch.optim", optimizer["class"])
    optimizer = optimizer_class(model.parameters(), **optimizer["params"])

    for epoch in range(nr_epochs):

        # Train model
        model.train()
        running_loss = 0.0
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

            running_loss += combined_loss.item()
            progress_bar.set_postfix({"loss": running_loss / (iteration + 1)})

        # Validate model
        model.eval()
        running_loss = 0.0
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
                running_loss += combined_loss.item()
                progress_bar.set_postfix({"loss": running_loss / (iteration + 1)})


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
        "params" in config
    ), f"params not defined in config {args.config}: {config.keys()}"
    train_graph_classifier(
        model_config=config["model"],
        data_config=config["data"],
        **config["params"],
    )
