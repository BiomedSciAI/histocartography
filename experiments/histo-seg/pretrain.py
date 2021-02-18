import datetime
import logging
from typing import Dict, Optional
from copy import deepcopy

import mlflow
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import trange

from logging_helper import (
    LoggingHelper,
    prepare_experiment,
    robust_mlflow,
    log_parameters,
)
from losses import get_loss, get_lr
from models import PatchTissueClassifier
from utils import dynamic_import_from, get_config
from test_cnn import test_cnn, fill_missing_information
from train_utils import log_device, log_nr_parameters, get_optimizer


def train_patch_classifier(
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
    balanced_batches: bool = False,
    pretrain_epochs: Optional[int] = None,
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
    logging.info(f"Unmatched arguments for pretraining: {kwargs}")

    BACKGROUND_CLASS = dynamic_import_from(dataset, "BACKGROUND_CLASS")
    NR_CLASSES = dynamic_import_from(dataset, "NR_CLASSES")
    prepare_patch_datasets = dynamic_import_from(dataset, "prepare_patch_datasets")

    # Data loaders
    training_dataset, validation_dataset = prepare_patch_datasets(**data_config)
    if balanced_batches:
        training_sample_weights = training_dataset.get_class_weights()
        sampler = WeightedRandomSampler(
            training_sample_weights, len(training_dataset), replacement=True
        )
    else:
        sampler = None
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=not balanced_batches,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Compute device
    device = log_device()

    # Model
    model = PatchTissueClassifier(num_classes=NR_CLASSES, **model_config)
    model = model.to(device)
    log_nr_parameters(model)

    # Loss function
    criterion = get_loss(loss, device=device)

    training_metric_logger = LoggingHelper(
        metrics_config,
        prefix="train",
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
    )
    validation_metric_logger = LoggingHelper(
        metrics_config,
        prefix="valid",
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
    )

    # Optimizer
    optim, scheduler = get_optimizer(optimizer, model)

    if pretrain_epochs is not None:
        model.freeze_encoder()

    for epoch in trange(nr_epochs):

        # Train model
        time_before_training = datetime.datetime.now()
        model.train()
        if pretrain_epochs is not None and epoch == pretrain_epochs:
            model.unfreeze_encoder()

        for patches, labels in training_loader:
            patches = patches.to(device)
            labels = labels.to(device)

            optim.zero_grad()

            logits = model(patches)
            loss = criterion(logits, labels)

            loss.backward()
            if clip_gradient_norm is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), clip_gradient_norm
                )

            optim.step()

            training_metric_logger.add_iteration_outputs(
                loss=loss.item(),
                logits=logits.detach().cpu(),
                labels=labels.cpu(),
            )

        if scheduler is not None:
            robust_mlflow(mlflow.log_metric, "current_lr", get_lr(optim), epoch)
            scheduler.step()

        training_metric_logger.log_and_clear(epoch)
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
                for patches, labels in validation_loader:
                    patches = patches.to(device)
                    labels = labels.to(device)

                    logits = model(patches)
                    loss = criterion(logits, labels)

                    validation_metric_logger.add_iteration_outputs(
                        loss=loss.item(),
                        logits=logits.detach().cpu(),
                        labels=labels.cpu(),
                    )

            validation_metric_logger.log_and_clear(
                epoch, model=model if not test else None
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


def run_test(inference_mode="patch_based"):
    train_config = deepcopy(config)
    test_config_ = deepcopy(test_config)
    test_config_["params"]["inference_mode"] = inference_mode
    prepare_experiment(
        config_path=test_config_path,
        data={},
        model=test_config_["model"],
        params=test_config_["params"],
    )
    log_parameters(
        data=train_config["data"],
        model=train_config["model"],
        params=train_config["params"],
    )
    test_cnn(
        model_config=test_config_["model"],
        data_config=test_config_["data"],
        test=test,
        **test_config_["params"],
    )
    robust_mlflow(mlflow.end_run)


if __name__ == "__main__":
    config, config_path, test = get_config(
        name="train",
        default="config/pretrain.yml",
        required=("model", "data", "metrics", "params"),
    )
    logging.info("Start pre-training")
    tags = config["params"].get("experiment_tags", None)
    if test:
        config["data"]["overfit_test"] = True
        config["params"]["num_workers"] = 0
    prepare_experiment(config_path=config_path, **config)
    train_patch_classifier(
        model_config=config["model"],
        data_config=config["data"],
        metrics_config=config["metrics"],
        config_path=config_path,
        test=test,
        **config["params"],
    )

    # Automatically run testing code
    if config["params"].get("autotest", False):
        # End training run
        run_id = robust_mlflow(mlflow.active_run).info.run_id
        experiment_id = robust_mlflow(mlflow.active_run).info.experiment_id
        model_uri = f"s3://mlflow/{experiment_id}/{run_id}/artifacts/best.valid.MultiLabelBalancedAccuracy"
        robust_mlflow(mlflow.end_run)

        # Start testing run
        logging.info("Start testing")
        test_config, test_config_path, test = get_config(
            name="test",
            default=config_path,
            required=("model", "data"),
        )
        config["model"].pop("architecture")
        test_config["params"]["experiment_tags"] = tags  # Use same tags as for training
        test_config["model"]["architecture"] = model_uri  # Use best model from training
        fill_missing_information(test_config["model"], test_config["data"])

        run_test(inference_mode="patch_based")
        run_test(inference_mode="hacky")
