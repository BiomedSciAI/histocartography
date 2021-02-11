from logging_helper import robust_mlflow
import mlflow
from utils import dynamic_import_from
import torch
import dgl
from utils import get_config
import logging
from logging_helper import prepare_experiment, log_parameters
from test_gnn import test_gnn, fill_missing_information
from models import (
    SemiSuperPixelTissueClassifier,
    ImageTissueClassifier,
    SuperPixelTissueClassifier,
)


def prepare_training(default="default.yml"):
    config, config_path, test = get_config(
        name="train",
        default=f"config/{default}",
        required=("model", "data", "metrics", "params"),
    )
    logging.info("Start training")
    tags = config["params"].get("experiment_tags", None)
    prepare_experiment(config_path=config_path, **config)
    config["params"]["config_path"] = config_path
    config["params"]["test"] = test
    if test:
        config["data"]["overfit_test"] = True
        config["params"]["num_workers"] = 0
    return config, tags


def end_run():
    active_run = robust_mlflow(mlflow.active_run)
    if active_run is not None:
        run_id = active_run.info.run_id
        experiment_id = robust_mlflow(mlflow.active_run).info.experiment_id
        robust_mlflow(mlflow.end_run)
        return experiment_id, run_id
    return None, None


def auto_test(config, tags, model_uri, default="default.yml"):
    # Automatically run testing code
    if not config["params"].get("test", False) and config["params"].get(
        "autotest", False
    ):
        # Start testing run
        logging.info("Start testing")
        test_config, test_config_path, test = get_config(
            name="test",
            default=f"config/{default}",
            required=("model", "data"),
        )

        test_config["params"]["experiment_tags"] = tags  # Use same tags as for training
        test_config["model"]["architecture"] = model_uri  # Use best model from training
        fill_missing_information(test_config["model"], test_config["data"])
        prepare_experiment(config_path=test_config_path, **test_config)
        log_parameters(
            data=config["data"], model=config["model"], params=config["params"]
        )
        test_gnn(
            model_config=test_config["model"],
            data_config=test_config["data"],
            test=test,
            **test_config["params"],
        )


def log_nr_parameters(model):
    nr_trainable_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    robust_mlflow(mlflow.log_param, "nr_parameters", nr_trainable_total_params)
    if isinstance(model, ImageTissueClassifier):
        mode = "weak_supervision"
    elif isinstance(model, SuperPixelTissueClassifier):
        mode = "strong_supervision"
    elif isinstance(model, SemiSuperPixelTissueClassifier):
        mode = "combined_supervision"
    else:
        mode = "unknown"
    robust_mlflow(mlflow.log_param, "supervision", mode)


def get_optimizer(optimizer, model):
    optimizer_class = dynamic_import_from("torch.optim", optimizer["class"])
    optim = optimizer_class(model.parameters(), **optimizer["params"])

    # Learning rate scheduler
    scheduler_config = optimizer.get("scheduler", None)
    if scheduler_config is not None:
        scheduler_class = dynamic_import_from(
            "torch.optim.lr_scheduler", scheduler_config["class"]
        )
        scheduler = scheduler_class(optim, **scheduler_config.get("params", {}))
    else:
        scheduler = None
    return optim, scheduler


def log_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        robust_mlflow(mlflow.log_param, "device", torch.cuda.get_device_name(0))
    else:
        robust_mlflow(mlflow.log_param, "device", "CPU")
    return device
