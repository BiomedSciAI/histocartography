from utils import dynamic_import_from
from typing import Dict, NewType
import logging
import torch
import mlflow
from pathlib import Path
from tqdm.auto import tqdm
import datetime
from collections import defaultdict

from utils import get_config
from logging_helper import prepare_experiment, robust_mlflow
from inference import PatchBasedInference
from metrics import IoU, MeanIoU, fIoU, F1Score, MeanF1Score


def get_model(architecture):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    if architecture.startswith("s3://mlflow"):
        model = mlflow.pytorch.load_model(architecture, map_location=device)
    elif architecture.endswith(".pth"):
        model = torch.load(architecture, map_location=device)
    else:
        model_class = dynamic_import_from("torchvision.models", architecture)
        model = model_class(pretrained=True)
    return model


def parse_mlflow_list(x: str):
    return list(map(float, x[1:-1].split(",")))


def fill_missing_information(model_config, data_config):
    if model_config["architecture"].startswith("s3://mlflow"):
        _, _, _, experiment_id, run_id, _, _ = model_config["architecture"].split("/")
        df = mlflow.search_runs(experiment_id)
        df = df.set_index("run_id")

        if "downsample_factor" not in data_config:
            data_config["downsample_factor"] = float(
                df.loc[run_id, "params.data.downsample_factor"]
            )
        if "image_path" not in data_config:
            data_config["image_path"] = df.loc[run_id, "params.data.image_path"]
        if "normalizer" not in data_config:
            normalizer = dict()
            normalizer["type"] = df.loc[run_id, "params.data.normalizer.type"]
            normalizer["mean"] = parse_mlflow_list(
                df.loc[run_id, "params.data.normalizer.mean"]
            )
            normalizer["std"] = parse_mlflow_list(
                df.loc[run_id, "params.data.normalizer.std"]
            )
            data_config["normalizer"] = normalizer


def test_cnn(
    dataset: str,
    data_config: Dict,
    model_config: Dict,
    patch_size: int,
    overlap: int,
    test: bool,
    operation: str,
    local_save_path,
    mlflow_save_path,
    **kwargs,
):
    logging.info(f"Unmatched arguments for testing: {kwargs}")

    NR_CLASSES = dynamic_import_from(dataset, "NR_CLASSES")
    BACKGROUND_CLASS = dynamic_import_from(dataset, "BACKGROUND_CLASS")
    prepare_patch_testset = dynamic_import_from(dataset, "prepare_patch_testset")
    show_class_acivation = dynamic_import_from(dataset, "show_class_acivation")
    show_segmentation_masks = dynamic_import_from(dataset, "show_segmentation_masks")

    fill_missing_information(model_config, data_config)
    test_dataset = prepare_patch_testset(test=test, **data_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        robust_mlflow(mlflow.log_param, "device", torch.cuda.get_device_name(0))
    else:
        robust_mlflow(mlflow.log_param, "device", "CPU")

    model = get_model(**model_config)
    model = model.to(device)

    inferencer = PatchBasedInference(
        model=model,
        device=device,
        nr_classes=NR_CLASSES,
        patch_size=(patch_size, patch_size),
        overlap=(overlap, overlap),
        **kwargs,
    )
    run_id = robust_mlflow(mlflow.active_run).info.run_id

    local_save_path = Path(local_save_path)
    if not local_save_path.exists():
        local_save_path.mkdir()
    save_path = local_save_path / run_id
    if not save_path.exists():
        save_path.mkdir()

    metrics = [
        c(nr_classes=NR_CLASSES, background_label=BACKGROUND_CLASS)
        for c in [IoU, F1Score, MeanF1Score, MeanIoU, fIoU]
    ]
    results = defaultdict(list)

    for i, (name, image, annotation, annotation2) in enumerate(tqdm(test_dataset)):
        time_before = datetime.datetime.now()
        predicted_mask = inferencer.predict(image, operation=operation)

        for metric in tqdm(metrics, desc="Compute metrics"):
            for pathologist, ground_truth in zip([1, 2], [annotation, annotation2]):
                results[f"{metric.__class__.__name__}_{pathologist}"].append(
                    metric(
                        prediction=predicted_mask.unsqueeze(0),
                        ground_truth=ground_truth.unsqueeze(0),
                    )
                    .squeeze(0)
                    .tolist()
                )

        if i > 0 and i % 10 == 0:
            for k, v in tqdm(results.items(), desc="Log metrics"):
                values = torch.Tensor(v)
                if len(values.shape) == 1:
                    robust_mlflow(mlflow.log_metric, k, values.mean().item(), i)
                else:
                    mask = torch.isnan(values)
                    values[mask] = 0
                    means = torch.sum(values, axis=0) / torch.sum(~mask, axis=0)
                    for j, mean in enumerate(means):
                        robust_mlflow(
                            mlflow.log_metric, f"{k}_class_{j}", mean.item(), i
                        )

        # Save figure
        if operation == "per_class":
            fig = show_class_acivation(predicted_mask)
        elif operation == "argmax":
            fig = show_segmentation_masks(
                predicted_mask, annotation=annotation, annotation2=annotation2
            )
        else:
            raise NotImplementedError(
                f"Only support operation [per_class, argmax], but got {operation}"
            )
        file_name = save_path / f"{name}.png"
        fig.savefig(str(file_name), dpi=300, bbox_inches="tight")
        if mlflow_save_path is not None:
            robust_mlflow(
                mlflow.log_artifact,
                str(file_name),
                artifact_path=f"test_{operation}_segmentation_maps",
            )

        # Log duration
        duration = (datetime.datetime.now() - time_before).total_seconds()
        robust_mlflow(
            mlflow.log_metric,
            "seconds_per_image",
            duration,
            step=i,
        )


if __name__ == "__main__":
    config, config_path, test = get_config(
        name="test",
        default="pretrain.yml",
        required=("model", "data"),
    )
    prepare_experiment(config_path=config_path, **config)
    test_cnn(
        model_config=config["model"],
        data_config=config["data"],
        test=test,
        **config["params"],
    )
