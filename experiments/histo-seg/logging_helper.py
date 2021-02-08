import logging
from metrics import Metric
import os
import tempfile
import time
from http.client import RemoteDisconnected
from pathlib import Path
from typing import DefaultDict, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import yaml
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


def log_dependencies():
    with tempfile.TemporaryDirectory() as temp_dir_name:
        os.system(
            f'conda env export | grep -v "^prefix: " > {temp_dir_name}/environment.yml'
        )
        robust_mlflow(
            mlflow.log_artifact, f"{temp_dir_name}/environment.yml", "sources"
        )


def log_segmentation_results(results, step):
    for k, v in results.items():
        values = torch.Tensor(v)
        if len(values.shape) == 1:
            robust_mlflow(mlflow.log_metric, k, values.mean().item(), step)
        else:
            mask = torch.isnan(values)
            values[mask] = 0
            means = torch.sum(values, axis=0) / torch.sum(~mask, axis=0)
            for j, mean in enumerate(means):
                robust_mlflow(mlflow.log_metric, f"{k}_class_{j}", mean.item(), step)


def log_preprocessing_parameters(graph_path: Path):
    config_path = graph_path / "config.yml"
    if config_path.exists():
        with open(config_path, "r") as file:
            config = yaml.load(file)
            for i, stage in enumerate(config.get("stages", [])):
                for stage_name, v in stage.items():
                    if stage_name in ["io"]:
                        continue
                    else:
                        robust_mlflow(mlflow.log_param, f"{i}_{stage_name}", v["class"])
                        robust_mlflow(
                            mlflow.log_params,
                            flatten(
                                v.get("params", {}), parent_key=f"{i}_{stage_name}"
                            ),
                        )


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
    model: dict, data: dict, params: dict, config_path: Optional[str] = None, **kwargs
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
    if config_path is not None:
        robust_mlflow(mlflow.log_artifact, config_path, "config")
    log_sources()
    log_dependencies()

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
            else:
                logits = torch.as_tensor(logits)
            self.logits.append(logits)
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu()
            else:
                labels = torch.as_tensor(labels)
            self.labels.append(labels)
        for name, value in kwargs.items():
            self.extra_info[name].extend(value)

    def _log(self, name, value, step):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().mean().item()
        try:
            robust_mlflow(mlflow.log_metric, f"{self.prefix}.{name}", value, step)
        except mlflow.exceptions.MlflowException as e:
            print(f"Could not log {self.prefix}.{name} at {step}: {value}")
            raise e

    def _log_metrics(self, step):
        if len(self.logits) == 0 or len(self.labels) == 0:
            return list()
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        metric_values = list()

        name: str
        metric: Metric
        for name, metric in zip(self.metric_names, self.metrics):
            metric_value = metric(logits, labels, **self.extra_info)
            if metric.is_per_class:
                for i in range(len(metric_value)):
                    self._log(f"{name}_class_{i}", metric_value[i], step)
                mean_metric_value = np.nanmean(metric_value)
                self._log(f"Mean{name}", mean_metric_value, step)
                metric_values.append(mean_metric_value)
            else:
                self._log(name, metric_value, step)
                metric_values.append(metric_value)
        return metric_values

    def _log_loss(self, step):
        mean_loss = np.mean(self.losses)
        self._log("loss", mean_loss, step)
        return mean_loss

    def log_and_clear(self, step=None, model=None):
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
            if step is not None:
                all_information = zip(
                    self.metric_names,
                    self.metrics,
                    self.best_metric_values,
                    current_values,
                )
                metric: Metric
                for i, (name, metric, best_value, current_value) in enumerate(
                    all_information
                ):
                    if metric.is_better(current_value, best_value):
                        self._log(f"best.{name}", current_value, step)
                        if metric.logs_model and model is not None:
                            if metric.is_per_class:
                                name = f"Mean{name}"
                            robust_mlflow(
                                mlflow.pytorch.log_model,
                                model,
                                f"best.{self.prefix}.{name}",
                            )
                        self.best_metric_values[i] = current_value
        self._reset_epoch_stats()


class SegmentationLoggingHelper(LoggingHelper):
    def __init__(
        self, metrics_config, prefix, background_label, leading_zeros=6, **kwargs
    ) -> None:
        kwargs["background_label"] = background_label
        super().__init__(metrics_config, prefix=prefix, **kwargs)
        self.cmap = ListedColormap(["green", "blue", "yellow", "red", "white"])
        self.leading_zeros = leading_zeros
        self.background_label = background_label

    def _reset_epoch_stats(self):
        self.masks = list()
        super()._reset_epoch_stats()

    def add_iteration_outputs(self, logits, labels, mask=None, loss=None, **kwargs):
        assert logits.shape == labels.shape, f"{logits.shape}, {labels.shape}"
        assert labels.shape == mask.shape, f"{labels.shape}, {mask.shape}"
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu()
            self.masks.append(mask)
        return super().add_iteration_outputs(
            loss=loss,
            logits=logits,
            labels=labels,
            tissue_mask=list(mask) if mask is not None else None,
            **kwargs,
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
        if len(self.labels) > 0:
            random_batch = np.random.randint(low=0, high=len(self.labels))
            annotations = self.labels[random_batch]
            segmentation_maps = self.logits[random_batch]
            if len(self.masks) > 0:
                segmentation_maps[
                    torch.as_tensor(~(self.masks[random_batch].astype(bool)))
                ] = self.background_label
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

    def log_and_clear(self, step, model):
        self.save_segmentation_masks(step)
        return super().log_and_clear(step, model=model)


class GraphLoggingHelper:
    def __init__(self, name, metrics_config, prefix, **kwargs) -> None:
        self.classification_helper = LoggingHelper(
            metrics_config.get(name, {}), f"{prefix}.{name}", **kwargs
        )
        self.segmentation_helper = SegmentationLoggingHelper(
            metrics_config.get("segmentation", {}),
            f"{prefix}.{name}.segmentation",
            **kwargs,
        )

    def add_iteration_outputs(
        self,
        loss=None,
        logits=None,
        targets=None,
        annotation=None,
        predicted_segmentation=None,
        tissue_masks=None,
        **kwargs,
    ):
        self.classification_helper.add_iteration_outputs(
            loss=loss, logits=logits, labels=targets, **kwargs
        )
        if (
            predicted_segmentation is not None
            and annotation is not None
            and tissue_masks is not None
        ):
            self.segmentation_helper.add_iteration_outputs(
                logits=predicted_segmentation, labels=annotation, mask=tissue_masks
            )

    def log_and_clear(self, step, model=None):
        self.classification_helper.log_and_clear(step, model)
        self.segmentation_helper.log_and_clear(step, model)
