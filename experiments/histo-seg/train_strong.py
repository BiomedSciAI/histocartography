import datetime
import logging
from typing import Dict, Optional

import mlflow
from numpy.lib.arraysetops import isin
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm.auto import trange
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import GraphBatch, GraphClassificationDataset, collate_graphs
from logging_helper import (
    GraphLoggingHelper,
    robust_mlflow,
)
from losses import get_loss, get_lr
from models import SuperPixelTissueClassifier
from utils import dynamic_import_from, get_batched_segmentation_maps
from train_utils import (
    end_run,
    log_nr_parameters,
    get_optimizer,
    log_device,
    prepare_training,
    auto_test,
)


def train_node_classifier(
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
    use_weighted_loss: bool = False,
    use_log_frequency_weights: bool = True,
    focused_metric: str = "fF1Score",
    balanced_sampling: bool = False,
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

    BACKGROUND_CLASS = dynamic_import_from(dataset, "BACKGROUND_CLASS")
    NR_CLASSES = dynamic_import_from(dataset, "NR_CLASSES")
    VARIABLE_SIZE = dynamic_import_from(dataset, "VARIABLE_SIZE")
    prepare_graph_datasets = dynamic_import_from(dataset, "prepare_graph_datasets")

    # Data loaders
    training_dataset: GraphClassificationDataset
    validation_dataset: GraphClassificationDataset
    training_dataset, validation_dataset = prepare_graph_datasets(**data_config)
    if balanced_sampling:
        training_sample_weights = training_dataset.get_graph_size_weights()
        sampler = WeightedRandomSampler(
            training_sample_weights, len(training_dataset), replacement=True
        )
    else:
        sampler = None
    assert training_dataset.mode == "tissue"
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=not balanced_sampling,
        sampler=sampler,
        collate_fn=collate_graphs,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1 if VARIABLE_SIZE else batch_size,
        collate_fn=collate_graphs,
        num_workers=num_workers,
    )

    # Logging
    training_metric_logger = GraphLoggingHelper(
        name="node",
        metrics_config=metrics_config,
        prefix="train",
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
    )
    validation_metric_logger = GraphLoggingHelper(
        name="node",
        metrics_config=metrics_config,
        prefix="valid",
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
    )

    # Compute device
    device = log_device()

    # Model
    model = SuperPixelTissueClassifier(nr_classes=NR_CLASSES, **model_config)
    model = model.to(device)
    log_nr_parameters(model)

    # Loss function
    if use_weighted_loss:
        loss["node"]['params']['weight'] = training_dataset.get_overall_loss_weights(log=use_log_frequency_weights)
    criterion = get_loss(loss, "node", device)

    # Optimizer
    optim, scheduler = get_optimizer(optimizer, model)

    for epoch in trange(nr_epochs):

        # Train model
        time_before_training = datetime.datetime.now()
        model.train()
        graph_batch: GraphBatch
        for graph_batch in training_loader:

            optim.zero_grad()

            graph = graph_batch.meta_graph.to(device)
            labels = graph_batch.node_labels.to(device)
            logits = model(graph)

            # Calculate loss
            loss_information = {
                "logits": logits,
                "targets": labels,
                "node_associations": graph.batch_num_nodes,
            }
            loss = criterion(**loss_information)
            loss.backward()

            # Optimize
            if clip_gradient_norm is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), clip_gradient_norm
                )
            optim.step()

            # Log to MLflow
            training_metric_logger.add_iteration_outputs(
                loss=loss.item(), **loss_information
            )

        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            robust_mlflow(mlflow.log_metric, "current_lr", get_lr(optim), epoch)
            scheduler.step()

        training_metric_logger.log_and_clear(step=epoch)
        training_epoch_duration = (
            datetime.datetime.now() - time_before_training
        ).total_seconds()
        robust_mlflow(
            mlflow.log_metric,
            "train.seconds_per_epoch",
            training_epoch_duration,
            step=epoch,
        )

        if epoch > 0 and epoch % validation_frequency == 0:
            # Validate model
            time_before_validation = datetime.datetime.now()
            model.eval()

            for graph_batch in validation_loader:

                with torch.no_grad():
                    graph = graph_batch.meta_graph.to(device)
                    labels = graph_batch.node_labels.to(device)
                    logits = model(graph)

                    # Calculate loss
                    loss_information = {
                        "logits": logits,
                        "targets": labels,
                        "node_associations": graph.batch_num_nodes,
                    }
                    loss = criterion(**loss_information)

                assert (
                    graph_batch.has_validation_information
                ), f"Graph batch does not have information necessary for validation"
                segmentation_maps = get_batched_segmentation_maps(
                    node_logits=loss_information["logits"],
                    node_associations=graph.batch_num_nodes,
                    superpixels=graph_batch.instance_maps,
                    NR_CLASSES=NR_CLASSES,
                )
                segmentation_maps = torch.as_tensor(segmentation_maps)

                validation_metric_logger.add_iteration_outputs(
                    loss=loss.item(),
                    annotation=torch.as_tensor(graph_batch.segmentation_masks),
                    predicted_segmentation=segmentation_maps,
                    tissue_masks=graph_batch.tissue_masks.astype(bool),
                    **loss_information,
                )

            current_metrics = validation_metric_logger.log_and_clear(
                step=epoch, model=model if not test else None
            )
            validation_epoch_duration = (
                datetime.datetime.now() - time_before_validation
            ).total_seconds()
            robust_mlflow(
                mlflow.log_metric,
                "valid.seconds_per_epoch",
                validation_epoch_duration,
                step=epoch,
            )

            if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                robust_mlflow(mlflow.log_metric, "current_lr", get_lr(optim), epoch)
                scheduler.step(current_metrics[focused_metric])

    mlflow.pytorch.log_model(model, "latest")


if __name__ == "__main__":
    config, tags = prepare_training(default="default_strong.yml")
    focused_metric = config["params"].get("focused_metric", "MeanIoU")
    train_node_classifier(
        model_config=config["model"],
        data_config=config["data"],
        metrics_config=config["metrics"],
        **config["params"],
    )
    experiment_id, run_id = end_run()
    checkpoint = f"best.valid.node.segmentation.{focused_metric}"
    model_uri = f"s3://mlflow/{experiment_id}/{run_id}/artifacts/{checkpoint}"
    auto_test(config, tags, default="default_strong.yml", model_uri=model_uri)
