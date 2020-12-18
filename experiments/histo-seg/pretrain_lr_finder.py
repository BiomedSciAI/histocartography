import logging
import os
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import torch
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder

from logging_helper import prepare_experiment, robust_mlflow
from losses import get_loss
from models import PatchTissueClassifier
from utils import dynamic_import_from, get_config


def find_learning_rate(
    dataset: str,
    model_config: Dict,
    data_config: Dict,
    batch_size: int,
    num_workers: int,
    optimizer: Dict,
    loss: Dict,
    **kwargs,
) -> None:
    """Find a suitable learning rate for a pretraining task

    Args:
        model_config (Dict): Configuration of the models (gnn and classifier)
        data_config (Dict): Configuration of the data (e.g. splits)
        batch_size (int): Batch size
        nr_epochs (int): Number of epochs to train
        optimizer (Dict): Configuration of the optimizer
    """
    logging.info(f"Unmatched arguments for pretraining lr finding: {kwargs}")

    BACKGROUND_CLASS = dynamic_import_from(dataset, "BACKGROUND_CLASS")
    NR_CLASSES = dynamic_import_from(dataset, "NR_CLASSES")
    prepare_patch_datasets = dynamic_import_from(dataset, "prepare_patch_datasets")

    # Data loaders
    training_dataset, validation_dataset = prepare_patch_datasets(**data_config)
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        robust_mlflow(mlflow.log_param, "device", torch.cuda.get_device_name(0))
    else:
        robust_mlflow(mlflow.log_param, "device", "CPU")

    # Model
    model = PatchTissueClassifier(num_classes=NR_CLASSES, **model_config)
    model = model.to(device)
    nr_trainable_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    robust_mlflow(mlflow.log_param, "nr_parameters", nr_trainable_total_params)

    # Loss function
    criterion = get_loss(loss, device=device)

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
        start_lr=1e-7,
        end_lr=1,
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
    config, config_path, test = get_config(
        name="train",
        default="pretrain.yml",
        required=("model", "data", "metrics", "params"),
    )
    logging.info("Start pre-training LR finding")
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
