import datetime
import logging
from typing import Dict, MutableSequence, Optional, List, no_type_check

import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange

from dataset import GraphBatch, collate_graphs, GraphClassificationDataset
from inference import GraphGradCAMBasedInference
from logging_helper import (
    GraphLoggingHelper,
    LoggingHelper,
    robust_mlflow,
)
from losses import get_loss, get_lr
from models import SemiSuperPixelTissueClassifier
from utils import dynamic_import_from, get_batched_segmentation_maps
from train_utils import (
    end_run,
    log_nr_parameters,
    get_optimizer,
    log_device,
    prepare_training,
    auto_test,
)

class CombinedCriterion(torch.nn.Module):
    def __init__(self, loss: dict, device) -> None:
        super().__init__()
        self.node_criterion = get_loss(loss, "node", device)
        self.graph_criterion = get_loss(loss, "graph", device)
        self.node_loss_weight = loss.get("node_weight", 0.5)
        assert (
            0.0 <= self.node_loss_weight <= 1.0
        ), f"Node weight loss must be between 0 and 1, but is {self.node_loss_weight}"
        self.graph_loss_weight = 1.0 - self.node_loss_weight
        self.device = device

    def forward(
        self,
        graph_logits: Optional[torch.Tensor] = None,
        graph_labels: Optional[torch.Tensor] = None,
        node_logits: Optional[torch.Tensor] = None,
        node_labels: Optional[torch.Tensor] = None,
        node_associations: Optional[List[int]] = None,
    ):
        assert (
            node_logits is not None and node_labels is not None
        ), "Cannot use combined criterion without node input"
        assert (
            graph_logits is not None and graph_labels is not None
        ), "Cannot use combined criterion without graph input"
        node_labels = node_labels.to(self.device)
        graph_labels = graph_labels.to(self.device)
        node_loss = self.node_criterion(
            logits=node_logits,
            targets=node_labels,
            node_associations=node_associations,
        )
        graph_loss = self.graph_criterion(logits=graph_logits, targets=graph_labels)
        combined_loss = (
            self.graph_loss_weight * graph_loss + self.node_loss_weight * node_loss
        )
        return combined_loss, graph_loss.detach().cpu(), node_loss.detach().cpu()


def train_graph_classifier(
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
    assert training_dataset.mode == "tissue"
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    training_graph_metric_logger = GraphLoggingHelper(
        name="graph",
        metrics_config=metrics_config,
        prefix="train",
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
    )
    training_node_metric_logger = GraphLoggingHelper(
        name="node",
        metrics_config=metrics_config,
        prefix="train",
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
    )
    training_combined_metric_logger = LoggingHelper(
        metrics_config={},
        prefix="train.combined",
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
    )
    validation_graph_metric_logger = GraphLoggingHelper(
        name="graph",
        metrics_config=metrics_config,
        prefix="valid",
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
    )
    validation_node_metric_logger = GraphLoggingHelper(
        name="node",
        metrics_config=metrics_config,
        prefix="valid",
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
    )
    validation_combined_metric_logger = LoggingHelper(
        metrics_config={},
        prefix="valid.combined",
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
    )

    # Compute device
    device = log_device()

    # Model
    model = SemiSuperPixelTissueClassifier(nr_classes=NR_CLASSES, **model_config)
    model = model.to(device)
    log_nr_parameters(model)

    # Loss function
    if use_weighted_loss:
        training_dataset.set_mode("tissue")
        loss["node"]['params']['weight'] = training_dataset.get_overall_loss_weights()
        training_dataset.set_mode("image")
        loss["graph"]['params']['weight'] = training_dataset.get_overall_loss_weights()
        training_dataset.set_mode("tissue")
    criterion = CombinedCriterion(loss, device)

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
            graph_logits, node_logits = model(graph)

            # Calculate loss
            loss_information = {
                "graph_logits": graph_logits,
                "graph_labels": graph_batch.graph_labels,
                "node_logits": node_logits,
                "node_labels": graph_batch.node_labels,
                "node_associations": graph.batch_num_nodes,
            }
            combined_loss, graph_loss, node_loss = criterion(**loss_information)
            combined_loss.backward()

            # Optimize
            if clip_gradient_norm is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), clip_gradient_norm
                )
            optim.step()

            # Log to MLflow
            training_graph_metric_logger.add_iteration_outputs(
                loss=graph_loss,
                logits=loss_information["graph_logits"],
                targets=loss_information["graph_labels"],
            )
            training_node_metric_logger.add_iteration_outputs(
                loss=node_loss,
                logits=loss_information["node_logits"],
                targets=loss_information["node_labels"],
                node_associations=graph.batch_num_nodes,
            )
            training_combined_metric_logger.add_iteration_outputs(loss=combined_loss)

        if scheduler is not None:
            robust_mlflow(mlflow.log_metric, "current_lr", get_lr(optim), epoch)
            scheduler.step()

        training_combined_metric_logger.log_and_clear(epoch)
        training_graph_metric_logger.log_and_clear(epoch)
        training_node_metric_logger.log_and_clear(epoch)
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
                    graph_logits, node_logits = model(graph)

                    # Calculate loss
                    loss_information = {
                        "graph_logits": graph_logits,
                        "graph_labels": graph_batch.graph_labels,
                        "node_logits": node_logits,
                        "node_labels": graph_batch.node_labels,
                        "node_associations": graph.batch_num_nodes,
                    }
                    combined_loss, graph_loss, node_loss = criterion(**loss_information)

                assert (
                    graph_batch.segmentation_masks is not None
                ), f"Cannot compute segmentation metrics if annotations are not loaded"

                # Graph Head Prediction
                inferencer = GraphGradCAMBasedInference(
                    NR_CLASSES, model, device=device
                )
                gradcam_segmentation_maps = inferencer.predict_batch(
                    graph, graph_batch.instance_maps
                )
                validation_graph_metric_logger.add_iteration_outputs(
                    loss=graph_loss,
                    logits=loss_information["graph_logits"],
                    targets=loss_information["graph_labels"],
                    annotation=torch.as_tensor(graph_batch.segmentation_masks),
                    predicted_segmentation=gradcam_segmentation_maps,
                    tissue_masks=graph_batch.tissue_masks.astype(bool),
                )

                # Node Head Prediction
                node_segmentation_maps = get_batched_segmentation_maps(
                    node_logits=loss_information["node_logits"],
                    node_associations=graph.batch_num_nodes,
                    superpixels=graph_batch.instance_maps,
                    NR_CLASSES=NR_CLASSES,
                )
                node_segmentation_maps = torch.as_tensor(node_segmentation_maps)
                validation_node_metric_logger.add_iteration_outputs(
                    loss=node_loss,
                    logits=loss_information["node_logits"],
                    targets=loss_information["node_labels"],
                    annotation=torch.as_tensor(graph_batch.segmentation_masks),
                    predicted_segmentation=node_segmentation_maps,
                    tissue_masks=graph_batch.tissue_masks.astype(bool),
                    node_associations=graph.batch_num_nodes,
                )

                validation_combined_metric_logger.add_iteration_outputs(
                    loss=combined_loss
                )

            validation_combined_metric_logger.log_and_clear(
                step=epoch, model=model if not test else None
            )
            validation_graph_metric_logger.log_and_clear(
                step=epoch, model=model if not test else None
            )
            validation_node_metric_logger.log_and_clear(
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

    mlflow.pytorch.log_model(model, "latest")


if __name__ == "__main__":
    config, tags = prepare_training(default="default_semi.yml")
    train_graph_classifier(
        model_config=config["model"],
        data_config=config["data"],
        metrics_config=config["metrics"],
        **config["params"],
    )
    experiment_id, run_id = end_run()
    checkpoint = "best.valid.node.segmentation.MeanIoU"
    model_uri = f"s3://mlflow/{experiment_id}/{run_id}/artifacts/{checkpoint}"
    auto_test(config, tags, default="default_strong.yml", model_uri=model_uri)
    end_run()

    checkpoint = "best.valid.graph.segmentation.MeanIoU"
    model_uri = f"s3://mlflow/{experiment_id}/{run_id}/artifacts/{checkpoint}"
    auto_test(config, tags, default="default_weak.yml", model_uri=model_uri)
