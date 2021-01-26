import datetime
import logging
from typing import Dict, List, Optional

import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange

from dataset import GraphBatch, collate_graphs
from logging_helper import (
    GraphClassificationLoggingHelper,
    prepare_experiment,
    robust_mlflow,
)
from losses import get_loss, get_lr
from models import (
    ImageTissueClassifier,
    SemiSuperPixelTissueClassifier,
    SuperPixelTissueClassifier,
)
from utils import dynamic_import_from, get_config, get_segmentation_map


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
        self.device = device

    def forward(
        self,
        graph_logits: Optional[torch.Tensor] = None,
        graph_labels: Optional[torch.Tensor] = None,
        node_logits: Optional[torch.Tensor] = None,
        node_labels: Optional[torch.Tensor] = None,
        node_associations: Optional[List[int]] = None,
    ):
        if self.mode == "strong_supervision":
            assert (
                node_logits is not None and node_labels is not None
            ), "Cannot use node criterion without input"
            return self.node_criterion(
                logits=node_logits,
                targets=node_labels.to(self.device),
                graph_associations=node_associations,
            )
        elif self.mode == "weak_supervision":
            assert (
                graph_logits is not None and graph_labels is not None
            ), "Cannot use graph criterion without input"
            return self.graph_criterion(
                logits=graph_logits, targets=graph_labels.to(self.device)
            )
        elif self.mode == "semi_strong_supervision":
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
                graph_associations=node_associations,
            )
            graph_loss = self.graph_criterion(logits=graph_logits, targets=graph_labels)
            combined_loss = (
                self.graph_loss_weight * graph_loss + self.node_loss_weight * node_loss
            )
            return combined_loss
        else:
            raise NotImplementedError(f"Criterion mode {self.mode} not implemented")


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
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
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

    # Consistency
    if mode in ["strong_supervision", "semi_strong_supervision"]:
        assert training_dataset.mode == "tissue"
    elif mode in ["weak_supervision"]:
        assert training_dataset.mode == "image"

    # Loss function
    criterion = CombinedCriterion(loss=loss, mode=mode, device=device)

    training_metric_logger = GraphClassificationLoggingHelper(
        metrics_config,
        "train",
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
    )
    validation_metric_logger = GraphClassificationLoggingHelper(
        metrics_config,
        "valid",
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
    )

    # Optimizer
    optimizer_class = dynamic_import_from("torch.optim", optimizer["class"])
    optim = optimizer_class(model.parameters(), **optimizer["params"])

    # Learning rate scheduler
    scheduler_config = optimizer.get("scheduler", None)
    if scheduler_config is not None:
        scheduler_class = dynamic_import_from(
            "torch.optim.lr_scheduler", scheduler_config["class"]
        )
        scheduler = scheduler_class(optim, **scheduler_config.get("params", {}))

    for epoch in trange(nr_epochs):

        # Train model
        time_before_training = datetime.datetime.now()
        model.train()
        graph_batch: GraphBatch
        for graph_batch in training_loader:
            assert (
                not (mode == "strong_supervision") or graph_batch.is_strongly_supervised
            )
            assert not (mode == "weak_supervision") or graph_batch.is_weakly_supervised

            optim.zero_grad()

            graph = graph_batch.meta_graph.to(device)
            logits = model(graph)

            # Calculate loss
            loss_information = get_logit_information(logits, mode)
            loss_information.update(
                {
                    "graph_labels": graph_batch.graph_labels,
                    "node_labels": graph_batch.node_labels,
                    "node_associations": graph.batch_num_nodes,
                }
            )
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

        if scheduler_config is not None:
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

        if epoch % validation_frequency == 0:
            # Validate model
            time_before_validation = datetime.datetime.now()
            model.eval()
            with torch.no_grad():
                for graph_batch in validation_loader:

                    graph = graph_batch.meta_graph.to(device)
                    logits = model(graph)

                    # Calculate loss
                    loss_information = get_logit_information(logits, mode)
                    loss_information.update(
                        {
                            "graph_labels": graph_batch.graph_labels,
                            "node_labels": graph_batch.node_labels,
                            "node_associations": graph.batch_num_nodes,
                        }
                    )
                    loss = criterion(**loss_information)

                    if mode in ["strong_supervision", "semi_strong_supervision"]:
                        assert (
                            graph_batch.has_validation_information
                        ), f"Graph batch does not have information necessary for validation"
                        segmentation_maps = get_segmentation_map(
                            node_logits=loss_information["node_logits"],
                            node_associations=graph.batch_num_nodes,
                            superpixels=graph_batch.instance_maps,
                            NR_CLASSES=NR_CLASSES,
                        )
                        annotations = torch.as_tensor(graph_batch.segmentation_masks)
                        segmentation_maps = torch.as_tensor(segmentation_maps)
                    else:
                        annotations = None
                        segmentation_maps = None

                    validation_metric_logger.add_iteration_outputs(
                        loss=loss.item(),
                        annotation=annotations,
                        predicted_segmentation=segmentation_maps,
                        tissue_masks=graph_batch.tissue_masks,
                        **loss_information,
                    )

            validation_metric_logger.log_and_clear(
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


if __name__ == "__main__":
    config, config_path, test = get_config(
        name="train",
        default="config/default.yml",
        required=("model", "data", "metrics", "params"),
    )
    logging.info("Start training")
    prepare_experiment(config_path=config_path, **config)
    train_graph_classifier(
        model_config=config["model"],
        data_config=config["data"],
        metrics_config=config["metrics"],
        config_path=config_path,
        test=test,
        **config["params"],
    )
