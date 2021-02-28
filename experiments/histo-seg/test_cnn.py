import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

from dataset import ImageDatapoint
from inference import ImageDatasetInference, ImageInferenceModel, PatchBasedInference
from logging_helper import LoggingHelper, prepare_experiment, robust_mlflow
from utils import dynamic_import_from, get_config


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
    threshold: float,
    operation: str,
    local_save_path: str,
    mlflow_save_path: str,
    inference_mode: str,
    **kwargs,
):
    logging.info(f"Unmatched arguments for testing: {kwargs}")

    NR_CLASSES = dynamic_import_from(dataset, "NR_CLASSES")
    BACKGROUND_CLASS = dynamic_import_from(dataset, "BACKGROUND_CLASS")
    prepare_patch_testset = dynamic_import_from(dataset, "prepare_patch_testset")
    show_class_acivation = dynamic_import_from(dataset, "show_class_acivation")
    show_segmentation_masks = dynamic_import_from(dataset, "show_segmentation_masks")

    test_dataset = prepare_patch_testset(test=test, **data_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        robust_mlflow(mlflow.log_param, "device", torch.cuda.get_device_name(0))
    else:
        robust_mlflow(mlflow.log_param, "device", "CPU")

    model = get_model(**model_config)
    model = model.to(device)

    if inference_mode == "patch_based":
        inferencer = PatchBasedInference(
            model=model,
            device=device,
            nr_classes=NR_CLASSES,
            patch_size=(patch_size, patch_size),
            overlap=(overlap, overlap),
            **kwargs,
        )
    else:
        inferencer = ImageInferenceModel(
            model=model,
            device=device,
            final_shape=test_dataset[0].segmentation_mask.shape,
        )

    run_id = robust_mlflow(mlflow.active_run).info.run_id

    local_save_path = Path(local_save_path)
    if not local_save_path.exists():
        local_save_path.mkdir()
    save_path = local_save_path / run_id
    if not save_path.exists():
        save_path.mkdir()

    logger_pathologist_1 = LoggingHelper(
        ["IoU", "F1Score", "GleasonScoreKappa", "GleasonScoreF1", "fIoU", "fF1Score", "DatasetDice", "DatasetIoU"],
        prefix="pathologist1",
        nr_classes=NR_CLASSES,
        background_label=BACKGROUND_CLASS,
    )
    logger_pathologist_2 = LoggingHelper(
        ["IoU", "F1Score", "GleasonScoreKappa", "GleasonScoreF1", "fIoU", "fF1Score", "DatasetDice", "DatasetIoU"],
        prefix="pathologist2",
        nr_classes=NR_CLASSES,
        background_label=BACKGROUND_CLASS,
    )

    def log_segmentation_mask(
        prediction: np.ndarray,
        datapoint: ImageDatapoint,
    ):
        tissue_mask = datapoint.tissue_mask
        prediction[~tissue_mask.astype(bool)] = BACKGROUND_CLASS

        # Save figure
        if operation == "per_class":
            fig = show_class_acivation(prediction)
        elif operation == "argmax":
            fig = show_segmentation_masks(
                prediction,
                annotation=datapoint.segmentation_mask,
                annotation2=datapoint.additional_segmentation_mask,
            )
        else:
            raise NotImplementedError(
                f"Only support operation [per_class, argmax], but got {operation}"
            )
        file_name = save_path / f"{datapoint.name}.png"
        fig.savefig(str(file_name), dpi=300, bbox_inches="tight")
        if mlflow_save_path is not None:
            robust_mlflow(
                mlflow.log_artifact,
                str(file_name),
                artifact_path=f"test_{operation}_segmentation_maps",
            )
        plt.close(fig=fig)

    inference_runner = ImageDatasetInference(
        inferencer=inferencer, callbacks=[log_segmentation_mask]
    )
    inference_runner(
        dataset=test_dataset,
        logger=logger_pathologist_1,
        additional_logger=logger_pathologist_2,
        operation=operation,
    )


if __name__ == "__main__":
    config, config_path, test = get_config(
        name="test",
        default="config/pretrain.yml",
        required=("model", "data"),
    )
    fill_missing_information(config["model"], config["data"])
    prepare_experiment(config_path=config_path, **config)
    test_cnn(
        model_config=config["model"],
        data_config=config["data"],
        test=test,
        **config["params"],
    )
