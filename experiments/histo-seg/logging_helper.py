from pathlib import Path
from typing import DefaultDict

import mlflow
import numpy as np
import torch

from utils import dynamic_import_from


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

    def add_iteration_outputs(self, losses, logits=None, labels=None, **kwargs):
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
        current_loss = self._log_loss(step)
        current_values = self._log_metrics(step)
        if current_loss < self.best_loss:
            self._log("best_loss", current_loss, step)
            self.best_loss = current_loss
            if model is not None:
                mlflow.pytorch.log_model(model, f"best.{self.prefix}.loss")
        all_information = zip(
            self.metric_names, self.metrics, self.best_metric_values, current_values
        )
        for i, (name, metric, best_value, current_value) in enumerate(all_information):
            if metric.is_better(current_value, best_value):
                self._log(f"best.{name}", current_value, step)
                if model is not None:
                    mlflow.pytorch.log_model(model, f"best.{self.prefix}.{name}")
                self.best_metric_values[i] = current_value

        self._reset_epoch_stats()


class GraphClassificationLoggingHelper:
    def __init__(self, metrics_config, prefix, **kwargs) -> None:
        self.graph_logger = LoggingHelper(
            metrics_config.get("graph", {}), f"{prefix}.graph", **kwargs
        )
        self.node_logger = LoggingHelper(
            metrics_config.get("node", {}), f"{prefix}.node", **kwargs
        )
        self.combined_logger = LoggingHelper({}, f"{prefix}.combined")

    def add_iteration_outputs(
        self,
        graph_loss,
        node_loss,
        graph_logits,
        node_logits,
        graph_labels,
        node_labels,
        node_associations,
    ):
        self.graph_logger.add_iteration_outputs(graph_loss, graph_logits, graph_labels)
        self.node_logger.add_iteration_outputs(
            node_loss, node_logits, node_labels, node_associations=node_associations
        )
        self.combined_logger.add_iteration_outputs(graph_loss + node_loss)

    def log_and_clear(self, step, model=None):
        self.graph_logger.log_and_clear(step, model)
        self.node_logger.log_and_clear(step, model)
        self.combined_logger.log_and_clear(step, model)
