import logging
import os
import shutil
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
    SCRATCH_PATH = Path(".")


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
    mlflow.log_params(flatten(data, "data"))
    mlflow.log_params(flatten(model, "model"))
    mlflow.log_params(flatten(kwargs))


def log_sources():
    for file in Path(".").iterdir():
        if file.name.endswith(".py"):
            mlflow.log_artifact(str(file), "sources")


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
    mlflow.log_artifact(config_path, "config")
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

    def add_iteration_outputs(self, losses=None, logits=None, labels=None, **kwargs):
        if losses is not None:
            self.losses.append(losses)
        if logits is not None:
            self.logits.append(logits)
        if labels is not None:
            self.labels.append(labels)
        for name, value in kwargs.items():
            self.extra_info[name].extend(value)

    def _log(self, name, value, step):
        mlflow.log_metric(f"{self.prefix}.{name}", value, step)

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
                    mlflow.pytorch.log_model(model, f"best.{self.prefix}.loss")
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
                        mlflow.pytorch.log_model(model, f"best.{self.prefix}.{name}")
                    self.best_metric_values[i] = current_value
        self._reset_epoch_stats()


class GraphClassificationLoggingHelper:
    def __init__(
        self, metrics_config, prefix, node_loss_weight, graph_loss_weight, **kwargs
    ) -> None:
        self.graph_logger = LoggingHelper(
            metrics_config.get("graph", {}), f"{prefix}.graph", **kwargs
        )
        self.node_logger = LoggingHelper(
            metrics_config.get("node", {}), f"{prefix}.node", **kwargs
        )
        self.node_loss_weight = node_loss_weight
        self.graph_loss_weight = graph_loss_weight
        self.combined_logger = LoggingHelper({}, f"{prefix}.combined")
        self.segmentation_logger = LoggingHelper(
            metrics_config.get("segmentation", {}), f"{prefix}.segmentation", **kwargs
        )
        self.cmap = ListedColormap(["green", "blue", "yellow", "red", "white"])

    def add_iteration_outputs(
        self,
        graph_loss=None,
        node_loss=None,
        graph_logits=None,
        node_logits=None,
        graph_labels=None,
        node_labels=None,
        node_associations=None,
        annotation=None,
        predicted_segmentation=None,
    ):
        if graph_loss is not None:
            self.graph_logger.add_iteration_outputs(
                graph_loss, graph_logits, graph_labels
            )
        if node_loss is not None:
            self.node_logger.add_iteration_outputs(
                node_loss, node_logits, node_labels, node_associations=node_associations
            )
        if graph_loss is not None and node_loss is not None:
            self.combined_logger.add_iteration_outputs(
                self.graph_loss_weight * graph_loss + self.node_loss_weight * node_loss
            )
        if annotation is not None and predicted_segmentation is not None:
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
            leading_zeros = (annotations.shape[0] // 10) + 1
            run_id = mlflow.active_run().info.run_id
            tmp_path = SCRATCH_PATH / run_id
            if not tmp_path.exists():
                tmp_path.mkdir()
            for i, (annotation, segmentation_map) in enumerate(
                zip(annotations, segmentation_maps)
            ):
                fig, _ = self.create_segmentation_maps(annotation, segmentation_map)
                name = (
                    tmp_path
                    / f"valid_segmap_epoch_{str(step).zfill(6)}_{str(i).zfill(leading_zeros)}.png"
                )
                fig.savefig(str(name))
                mlflow.log_artifact(
                    str(name), artifact_path="validation_segmentation_maps"
                )
                plt.close(fig=fig)
                plt.clf()
            shutil.rmtree(tmp_path)

    def log_and_clear(self, step, model=None):
        self.graph_logger.log_and_clear(step, model)
        self.node_logger.log_and_clear(step, model)
        self.combined_logger.log_and_clear(step, model)
        self.save_segmentation_masks(step)
        self.segmentation_logger.log_and_clear(step, model)
