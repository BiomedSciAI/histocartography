from functools import partial
import logging
from metrics import F1Score
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

from dataset import GraphDatapoint
from inference import (
    GraphBasedInference,
    GraphDatasetInference,
    GraphGradCAMBasedInference,
    GraphNodeBasedInference,
    NodeBasedInference,
    TTAGraphInference,
    AreaGraphCAMProbabilityBasedInference,
    AreaNodeProbabilityBasedInference,
)
from logging_helper import (
    LoggingHelper,
    log_device,
    prepare_experiment,
    robust_mlflow,
    log_confusion_matrix,
)
from models import (
    ImageTissueClassifier,
    SemiSuperPixelTissueClassifier,
    SuperPixelTissueClassifier,
)
from postprocess import create_dataset, run_mlp, train_mlp
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

        if (
            "use_augmentation_dataset" not in data_config
            and "params.data.use_augmentation_dataset" in df
        ):
            data_config["use_augmentation_dataset"] = (
                df.loc[run_id, "params.data.use_augmentation_dataset"] != "False"
            )
        if (
            "normalize" not in data_config
            and "params.data.normalize" in df
        ):
            data_config["normalize"] = (
                df.loc[run_id, "params.data.normalize"] != "False"
            )
        if "graph_directory" not in data_config:
            data_config["graph_directory"] = df.loc[
                run_id, "params.data.graph_directory"
            ]
        if "centroid_features" not in data_config:
            data_config["centroid_features"] = df.loc[
                run_id, "params.data.centroid_features"
            ]
        if (
            "normalize_features" not in data_config
            and "params.data.normalize_features" in df
        ):
            data_config["normalize_features"] = (
                df.loc[run_id, "params.data.normalize_features"] == "True"
            )
        if "fold" not in data_config and "params.data.fold" in df:
            data_config["fold"] = int(df.loc[run_id, "params.data.fold"])


def test_gnn(
    dataset: str,
    data_config: Dict,
    model_config: Dict,
    test: bool,
    operation: str,
    threshold: float,
    local_save_path,
    mlflow_save_path,
    use_grad_cam: bool = False,
    use_tta: bool = False,
    **kwargs,
):
    logging.info(f"Unmatched arguments for testing: {kwargs}")

    NR_CLASSES = dynamic_import_from(dataset, "NR_CLASSES")
    BACKGROUND_CLASS = dynamic_import_from(dataset, "BACKGROUND_CLASS")
    ADDITIONAL_ANNOTATION = dynamic_import_from(dataset, "ADDITIONAL_ANNOTATION")
    VARIABLE_SIZE = dynamic_import_from(dataset, "VARIABLE_SIZE")
    DISCARD_THRESHOLD = dynamic_import_from(dataset, "DISCARD_THRESHOLD")
    THRESHOLD = dynamic_import_from(dataset, "THRESHOLD")
    WSI_FIX = dynamic_import_from(dataset, "WSI_FIX")
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

    if isinstance(model, ImageTissueClassifier):
        mode = "weak_supervision"
    elif isinstance(model, SemiSuperPixelTissueClassifier):
        mode = "semi_strong_supervision"
    elif isinstance(model, SuperPixelTissueClassifier):
        mode = "strong_supervision"
    else:
        raise NotImplementedError

    if mode == "weak_supervision" or (
        mode == "semi_strong_supervision" and use_grad_cam
    ):
        inferencer = GraphGradCAMBasedInference(
            model=model, device=device, NR_CLASSES=NR_CLASSES, **kwargs
        )
        area_inferencer = AreaGraphCAMProbabilityBasedInference(
            model=model, device=device, NR_CLASSES=NR_CLASSES, **kwargs
        )
    else:
        inferencer = GraphNodeBasedInference(
            model=model,
            device=device,
            NR_CLASSES=NR_CLASSES,
            **kwargs,
        )
        area_inferencer = AreaNodeProbabilityBasedInference(
            model=model, device=device, **kwargs
        )

    run_id = robust_mlflow(mlflow.active_run).info.run_id

    local_save_path = Path(local_save_path)
    if not local_save_path.exists():
        local_save_path.mkdir()
    save_path = local_save_path / run_id
    if not save_path.exists():
        save_path.mkdir()

    logger_pathologist_1 = LoggingHelper(
        [
            "IoU",
            "F1Score",
            "GleasonScoreKappa",
            "GleasonScoreF1",
            "fIoU",
            "fF1Score",
            "DatasetDice",
            "DatasetIoU",
        ],
        prefix="pathologist1",
        nr_classes=NR_CLASSES,
        background_label=BACKGROUND_CLASS,
        variable_size=VARIABLE_SIZE,
        discard_threshold=DISCARD_THRESHOLD,
        threshold=THRESHOLD,
        wsi_fix=WSI_FIX,
        callbacks=[
            partial(
                log_confusion_matrix,
                classes=["Benign", "Grade6", "Grade7", "Grade8", "Grade9", "Grade10"],
                name="test.pathologist1.summed",
            )
        ],
    )
    if ADDITIONAL_ANNOTATION:
        logger_pathologist_2 = LoggingHelper(
            [
                "IoU",
                "F1Score",
                "GleasonScoreKappa",
                "GleasonScoreF1",
                "fIoU",
                "fF1Score",
                "DatasetDice",
                "DatasetIoU",
            ],
            prefix="pathologist2",
            nr_classes=NR_CLASSES,
            background_label=BACKGROUND_CLASS,
            variable_size=VARIABLE_SIZE,
            discard_threshold=DISCARD_THRESHOLD,
            threshold=THRESHOLD,
            wsi_fix=WSI_FIX,
            callbacks=[
                partial(
                    log_confusion_matrix,
                    classes=[
                        "Benign",
                        "Grade6",
                        "Grade7",
                        "Grade8",
                        "Grade9",
                        "Grade10",
                    ],
                    name="test.pathologist2.summed",
                )
            ],
        )
    else:
        logger_pathologist_2 = None

    def log_segmentation_mask(
        prediction: np.ndarray,
        datapoint: GraphDatapoint,
    ):
        tissue_mask = datapoint.tissue_mask
        prediction[~tissue_mask.astype(bool)] = BACKGROUND_CLASS
        ground_truth = datapoint.segmentation_mask.copy()
        ground_truth[~tissue_mask.astype(bool)] = BACKGROUND_CLASS
        if ADDITIONAL_ANNOTATION:
            ground_truth2 = datapoint.additional_segmentation_mask.copy()
            ground_truth2[~tissue_mask.astype(bool)] = BACKGROUND_CLASS
        else:
            ground_truth2 = None

        # Save figure
        if operation == "per_class":
            fig = show_class_acivation(prediction)
        elif operation == "argmax":
            fig = show_segmentation_masks(
                prediction,
                annotation=ground_truth,
                annotation2=ground_truth2,
            )
        else:
            raise NotImplementedError(
                f"Only support operation [per_class, argmax], but got {operation}"
            )

        # Set title to be DICE score
        metric = F1Score(
            nr_classes=NR_CLASSES,
            discard_threshold=DISCARD_THRESHOLD,
            background_label=BACKGROUND_CLASS,
        )
        metric_value = metric(prediction=[prediction], ground_truth=[ground_truth])
        fig.suptitle(
            f"Benign: {metric_value[0]}, Grade 3: {metric_value[1]}, Grade 4: {metric_value[2]}, Grade 5: {metric_value[3]}"
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

    if mode in ["strong_supervision", "semi_strong_supervision"]:
        classification_inferencer = NodeBasedInference(model=model, device=device)
        node_logger = LoggingHelper(
            ["NodeClassificationBalancedAccuracy", "NodeClassificationF1Score"],
            prefix="node",
            nr_classes=NR_CLASSES,
            background_label=BACKGROUND_CLASS,
        )
        classification_inferencer.predict(test_dataset, node_logger)
    if mode in ["weak_supervision", "semi_strong_supervision"]:
        classification_inferencer = GraphBasedInference(model=model, device=device)
        graph_logger = LoggingHelper(
            [
                "MultiLabelBalancedAccuracy",
                "MultiLabelF1Score",
                "GraphClassificationGleasonScoreKappa",
                "GraphClassificationGleasonScoreF1",
            ],
            prefix="graph",
            nr_classes=NR_CLASSES,
            background_label=BACKGROUND_CLASS,
            callbacks=[
                partial(
                    log_confusion_matrix,
                    classes=[
                        "Benign",
                        "Grade6",
                        "Grade7",
                        "Grade8",
                        "Grade9",
                        "Grade10",
                    ],
                    name="test.pathologist1.graph",
                )
            ],
        )
        classification_inferencer.predict(test_dataset, graph_logger)

    # Area Based Inference
    area_logger = LoggingHelper(
        ["AreaBasedGleasonScoreKappa", "AreaBasedGleasonScoreF1"],
        prefix="area",
        callbacks=[
            partial(
                log_confusion_matrix,
                classes=["Benign", "Grade6", "Grade7", "Grade8", "Grade9", "Grade10"],
                name="test.area.summed",
            )
        ],
        variable_size=VARIABLE_SIZE,
        wsi_fix=WSI_FIX,
    )
    inference_runner = GraphDatasetInference(inferencer=area_inferencer)
    inference_runner(
        dataset=test_dataset,
        logger=area_logger,
        operation=operation,
    )

    # Segmentation Inference
    if use_tta:
        inference_runner = TTAGraphInference(
            inferencer=inferencer,
            callbacks=[log_segmentation_mask],
            nr_classes=NR_CLASSES,
        )
    else:
        inference_runner = GraphDatasetInference(
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
        default="config/default_weak.yml",
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

    if config["params"].get("dataset", "eth") == "sicapv2_wsi":
        # Create percentage datasets
        training_dataset, validation_dataset, testing_dataset = create_dataset(
            model_config=config["model"],
            data_config=config["data"],
            test=test,
            **config["params"],
        )

        device = log_device()
        # Train MLP
        model = train_mlp(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            device=device,
            **config["params"],
        )

        # Evaluate on testset
        run_mlp(model=model, device=device, testing_dataset=testing_dataset)
