import logging
import os
import tempfile
import time
from http.client import RemoteDisconnected
from pathlib import Path
from typing import DefaultDict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from utils import dynamic_import_from, fix_seeds

with os.popen("hostname") as subprocess:
    hostname = subprocess.read()
if hostname.startswith("zhcc"):
    SCRATCH_PATH = Path("/dataL/anv/")
    if not SCRATCH_PATH.exists():
        try:
            SCRATCH_PATH.mkdir()
        except FileExistsError:
            pass
else:
    SCRATCH_PATH = Path("/tmp")


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            # Handle class names that are unnecessary
            if k == "class":
                new_key = parent_key

            # Handle lists (with no order)
            if isinstance(v, list):
                v = sorted(v)

            items.append((new_key, v))
    return dict(items)


def log_parameters(data, model, **kwargs):
    robust_mlflow(mlflow.log_params, flatten(data, "data"))
    robust_mlflow(mlflow.log_params, flatten(model, "model"))
    robust_mlflow(mlflow.log_params, flatten(kwargs))


def log_sources():
    for file in Path(".").iterdir():
        if file.name.endswith(".py"):
            robust_mlflow(mlflow.log_artifact, str(file), "sources")


def robust_mlflow(f, *args, max_tries=8, delay=1, backoff=2, **kwargs):
    while max_tries > 1:
        try:
            return f(*args, **kwargs)
        except RemoteDisconnected:
            print(f"MLFLOW remote disconnected. Trying again in {delay}s")
            time.sleep(delay)
            max_tries -= 1
            delay *= backoff
    return f(*args, **kwargs)


def prepare_experiment(
    model: dict, data: dict, params: dict, config_path: str, **kwargs
):
    logging.info(f"Unmatched arguments for MLflow logging: {kwargs}")

    seed = params.pop("seed", None)
    seed = fix_seeds(seed)

    # Basic MLflow setup
    mlflow.set_experiment(params.pop("experiment_name"))
    experiment_tags = params.pop("experiment_tags", None)
    if experiment_tags is not None:
        mlflow.set_tags(experiment_tags)

    # Artifacts
    robust_mlflow(mlflow.log_artifact, config_path, "config")
    log_sources()

    # Log everything relevant
    log_parameters(data, model, seed=seed, **params)


class LoggingHelper:
    def __init__(self, metrics_config, prefix="", **kwargs) -> None:
        self.metrics_config = metrics_config
        self.prefix = prefix
        self._reset_epoch_stats()
        self.best_loss = float("inf")
        self.metric_names = list()
        self.metrics = list()
        self.best_metric_values = list()
        for metric in metrics_config:
            metric_class = dynamic_import_from("metrics", metric)
            self.metrics.append(metric_class(**kwargs))
            self.metric_names.append(metric)
            self.best_metric_values.append(
                -float("inf") if metric_class.is_better(1, 0) else float("inf")
            )

    def _reset_epoch_stats(self):
        self.losses = list()
        self.logits = list()
        self.labels = list()
        self.extra_info = DefaultDict(list)

    def add_iteration_outputs(self, loss=None, logits=None, labels=None, **kwargs):
        if loss is not None:
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            self.losses.append(loss)
        if logits is not None:
            if isinstance(logits, torch.Tensor):
                logits = logits.detach().cpu()
            self.logits.append(logits)
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu()
            self.labels.append(labels)
        for name, value in kwargs.items():
            self.extra_info[name].extend(value)

    def _log(self, name, value, step):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().mean().item()
        robust_mlflow(mlflow.log_metric, f"{self.prefix}.{name}", value, step)

    def _log_metrics(self, step):
        if len(self.logits) == 0 or len(self.labels) == 0:
            return list()
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        metric_values = list()
        for name, metric in zip(self.metric_names, self.metrics):
            metric_value = metric(logits, labels, **self.extra_info)
            self._log(name, metric_value, step)
            metric_values.append(metric_value)
        return metric_values

    def _log_loss(self, step):
        mean_loss = np.mean(self.losses)
        self._log("loss", mean_loss, step)
        return mean_loss

    def log_and_clear(self, step, model=None):
        if len(self.losses) > 0:
            current_loss = self._log_loss(step)
            if current_loss < self.best_loss:
                self._log("best_loss", current_loss, step)
                self.best_loss = current_loss
                if model is not None:
                    robust_mlflow(
                        mlflow.pytorch.log_model, model, f"best.{self.prefix}.loss"
                    )
        if len(self.logits) > 0:
            current_values = self._log_metrics(step)
            all_information = zip(
                self.metric_names, self.metrics, self.best_metric_values, current_values
            )
            for i, (name, metric, best_value, current_value) in enumerate(
                all_information
            ):
                if metric.is_better(current_value, best_value):
                    self._log(f"best.{name}", current_value, step)
                    if model is not None:
                        robust_mlflow(
                            mlflow.pytorch.log_model,
                            model,
                            f"best.{self.prefix}.{name}",
                        )
                    self.best_metric_values[i] = current_value
        self._reset_epoch_stats()


class GraphClassificationLoggingHelper:
    def __init__(
        self, metrics_config, prefix, background_label, leading_zeros=6, **kwargs
    ) -> None:
        kwargs["background_label"] = background_label
        self.graph_logger = LoggingHelper(
            metrics_config.get("graph", {}), f"{prefix}.graph", **kwargs
        )
        self.node_logger = LoggingHelper(
            metrics_config.get("node", {}), f"{prefix}.node", **kwargs
        )
        self.combined_logger = LoggingHelper({}, f"{prefix}.loss")
        self.segmentation_logger = LoggingHelper(
            metrics_config.get("segmentation", {}), f"{prefix}.segmentation", **kwargs
        )
        self.cmap = ListedColormap(["green", "blue", "yellow", "red", "white"])
        self.prefix = prefix
        self.background_label = background_label
        self.leading_zeros = leading_zeros

    def add_iteration_outputs(
        self,
        loss=None,
        graph_logits=None,
        node_logits=None,
        graph_labels=None,
        node_labels=None,
        node_associations=None,
        annotation=None,
        predicted_segmentation=None,
    ):
        if graph_logits is not None and graph_labels is not None:
            self.graph_logger.add_iteration_outputs(
                loss=None, logits=graph_logits, labels=graph_labels
            )
        if node_logits is not None and node_labels is not None:
            self.node_logger.add_iteration_outputs(
                loss=None,
                logits=node_logits,
                labels=node_labels,
                node_associations=node_associations,
            )
        if loss is not None:
            self.combined_logger.add_iteration_outputs(loss=loss)
        if annotation is not None and predicted_segmentation is not None:
            predicted_segmentation[
                annotation == self.background_label
            ] = self.background_label
            self.segmentation_logger.add_iteration_outputs(
                logits=predicted_segmentation, labels=annotation
            )

    def create_segmentation_maps(self, actual, predicted):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax[0].imshow(
            actual, cmap=self.cmap, vmin=-0.5, vmax=4.5, interpolation="nearest"
        )
        ax[0].axis("off")
        ax[0].set_title("Ground Truth")
        ax[1].imshow(
            predicted, cmap=self.cmap, vmin=-0.5, vmax=4.5, interpolation="nearest"
        )
        ax[1].axis("off")
        ax[1].set_title(f"Prediction")
        return fig, ax

    def save_segmentation_masks(self, step):
        if len(self.segmentation_logger.labels) > 0:
            random_batch = np.random.randint(
                low=0, high=len(self.segmentation_logger.labels)
            )
            annotations = self.segmentation_logger.labels[random_batch]
            segmentation_maps = self.segmentation_logger.logits[random_batch]
            batch_leading_zeros = (annotations.shape[0] // 10) + 1
            run_id = robust_mlflow(mlflow.active_run).info.run_id
            with tempfile.TemporaryDirectory(
                prefix=run_id, dir=str(SCRATCH_PATH)
            ) as tmp_path:
                for i, (annotation, segmentation_map) in enumerate(
                    zip(annotations, segmentation_maps)
                ):
                    fig, _ = self.create_segmentation_maps(annotation, segmentation_map)
                    name = Path(tmp_path) / "{}_segmap_epoch_{}_{}.png".format(
                        self.prefix,
                        str(step).zfill(self.leading_zeros),
                        str(i).zfill(batch_leading_zeros),
                    )
                    fig.savefig(str(name), bbox_inches="tight")
                    robust_mlflow(
                        mlflow.log_artifact,
                        str(name),
                        artifact_path=f"{self.prefix}_segmentation_maps",
                    )
                    plt.close(fig=fig)
                    plt.clf()

    def log_and_clear(self, step, model=None):
        self.graph_logger.log_and_clear(step, model)
        self.node_logger.log_and_clear(step, model)
        self.combined_logger.log_and_clear(step, model)
        self.save_segmentation_masks(step)
        self.segmentation_logger.log_and_clear(step, model)
