import datetime
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import torch
from sklearn.metrics import cohen_kappa_score
from tqdm.auto import tqdm

from dataset import GraphDatapoint
from inference import GraphNodeBasedInference
from logging_helper import log_segmentation_results, prepare_experiment, robust_mlflow
from metrics import F1Score, IoU, MeanF1Score, MeanIoU, fIoU, sum_up_gleason
from utils import dynamic_import_from, get_config


def get_model(architecture):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    if architecture.startswith("s3://mlflow"):
        model = mlflow.pytorch.load_model(architecture, map_location=device)
    elif architecture.endswith(".pth"):
        model = torch.load(architecture, map_location=device)
    else:
        raise NotImplementedError(f"Cannot test on this architecture: {architecture}")
    return model


def parse_mlflow_list(x: str):
    return list(map(float, x[1:-1].split(",")))


def fill_missing_information(model_config, data_config):
    if model_config["architecture"].startswith("s3://mlflow"):
        _, _, _, experiment_id, run_id, _, _ = model_config["architecture"].split("/")
        df = mlflow.search_runs(experiment_id)
        df = df.set_index("run_id")

        if "use_augmentation_dataset" not in data_config:
            data_config["use_augmentation_dataset"] = (
                df.loc[run_id, "params.data.use_augmentation_dataset"] == "True"
            )
        if "graph_directory" not in data_config:
            data_config["graph_directory"] = df.loc[
                run_id, "params.data.graph_directory"
            ]
        if "centroid_features" not in data_config:
            data_config["centroid_features"] = df.loc[
                run_id, "params.data.centroid_features"
            ]
        if "normalize_features" not in data_config:
            data_config["normalize_features"] = (
                df.loc[run_id, "params.data.normalize_features"] == "True"
            )


def test_gnn(
    dataset: str,
    data_config: Dict,
    model_config: Dict,
    test: bool,
    operation: str,
    threshold: float,
    local_save_path,
    mlflow_save_path,
    **kwargs,
):
    logging.info(f"Unmatched arguments for testing: {kwargs}")

    NR_CLASSES = dynamic_import_from(dataset, "NR_CLASSES")
    BACKGROUND_CLASS = dynamic_import_from(dataset, "BACKGROUND_CLASS")
    prepare_graph_testset = dynamic_import_from(dataset, "prepare_graph_testset")
    show_class_acivation = dynamic_import_from(dataset, "show_class_acivation")
    show_segmentation_masks = dynamic_import_from(dataset, "show_segmentation_masks")

    test_dataset = prepare_graph_testset(test=test, **data_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        robust_mlflow(mlflow.log_param, "device", torch.cuda.get_device_name(0))
    else:
        robust_mlflow(mlflow.log_param, "device", "CPU")

    model = get_model(**model_config)
    model = model.to(device)

    inferencer = GraphNodeBasedInference(
        model=model,
        device=device,
        NR_CLASSES=NR_CLASSES,
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

    gleason_score_predictions = list()
    gleason_grade_1 = list()
    gleason_grade_2 = list()

    for i in tqdm(range(len(test_dataset))):
        graph_datapoint: GraphDatapoint = test_dataset[i]
        assert (
            graph_datapoint.has_validation_information
        ), f"Datapoint does not have validation information: {graph_datapoint}"
        assert (
            graph_datapoint.has_multiple_annotations
        ), f"Datapoint does not have multiple annotations: {graph_datapoint}"
        assert (
            graph_datapoint.name is not None
        ), f"Cannot test unnamed datapoint: {graph_datapoint}"

        time_before = datetime.datetime.now()

        predicted_mask = inferencer.predict(
            graph_datapoint.graph, graph_datapoint.instance_map, operation=operation
        )

        for metric in metrics:
            for pathologist, ground_truth in zip(
                [1, 2],
                [
                    graph_datapoint.segmentation_mask,
                    graph_datapoint.additional_segmentation_mask,
                ],
            ):
                results[f"{metric.__class__.__name__}_{pathologist}"].append(
                    metric(
                        prediction=predicted_mask.unsqueeze(0),
                        ground_truth=torch.as_tensor(ground_truth).unsqueeze(0),
                    )
                    .squeeze(0)
                    .tolist()
                )

        predicted_mask[~graph_datapoint.tissue_mask.astype(bool)] = BACKGROUND_CLASS
        gleason_score_predictions.append(
            sum_up_gleason(predicted_mask, n_class=NR_CLASSES, thres=threshold)
        )
        gleason_grade_1.append(
            sum_up_gleason(
                graph_datapoint.segmentation_mask, n_class=NR_CLASSES, thres=0
            )
        )
        gleason_grade_2.append(
            sum_up_gleason(
                graph_datapoint.additional_segmentation_mask,
                n_class=NR_CLASSES,
                thres=0,
            )
        )

        if i > 0 and i % 10 == 0:
            results["GleasonScoreKappa_1"] = [
                cohen_kappa_score(
                    gleason_grade_1, gleason_score_predictions, weights="quadratic"
                )
            ]
            results["GleasonScoreKappa_2"] = [
                cohen_kappa_score(
                    gleason_grade_2, gleason_score_predictions, weights="quadratic"
                )
            ]
            results["GleasonScoreKappa_inter"] = [
                cohen_kappa_score(gleason_grade_1, gleason_grade_2, weights="quadratic")
            ]
            log_segmentation_results(results, step=i)

        # Save figure
        if operation == "per_class":
            fig = show_class_acivation(predicted_mask)
        elif operation == "argmax":
            fig = show_segmentation_masks(
                predicted_mask,
                annotation=graph_datapoint.segmentation_mask,
                annotation2=graph_datapoint.additional_segmentation_mask,
            )
        else:
            raise NotImplementedError(
                f"Only support operation [per_class, argmax], but got {operation}"
            )
        file_name = save_path / f"{graph_datapoint.name}.png"
        fig.savefig(str(file_name), dpi=300, bbox_inches="tight")
        if mlflow_save_path is not None:
            robust_mlflow(
                mlflow.log_artifact,
                str(file_name),
                artifact_path=f"test_{operation}_segmentation_maps",
            )
        plt.close(fig=fig)

        # Log duration
        duration = (datetime.datetime.now() - time_before).total_seconds()
        robust_mlflow(
            mlflow.log_metric,
            "seconds_per_image",
            duration,
            step=i,
        )
    results["GleasonScoreKappa_1"] = [
        cohen_kappa_score(
            gleason_grade_1, gleason_score_predictions, weights="quadratic"
        )
    ]
    results["GleasonScoreKappa_2"] = [
        cohen_kappa_score(
            gleason_grade_2, gleason_score_predictions, weights="quadratic"
        )
    ]
    results["GleasonScoreKappa_inter"] = [
        cohen_kappa_score(gleason_grade_1, gleason_grade_2, weights="quadratic")
    ]
    log_segmentation_results(results, step=len(test_dataset))


if __name__ == "__main__":
    config, config_path, test = get_config(
        name="test",
        default="config/default.yml",
        required=("model", "data"),
    )
    fill_missing_information(config["model"], config["data"])
    prepare_experiment(config_path=config_path, **config)
    test_gnn(
        model_config=config["model"],
        data_config=config["data"],
        test=test,
        **config["params"],
    )
