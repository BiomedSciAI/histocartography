import datetime
from pathlib import Path

import cv2
import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import GraphClassificationDataset
from eth import ANNOTATIONS_DF, BACKGROUND_CLASS, IMAGES_DF, NR_CLASSES, PREPROCESS_PATH
from metrics import MeanIoU
from train import collate_valid, collate
from utils import merge_metadata


def compute_segmentation_map_guillaume(superpixel, node_prediction):
    segmentation_map = np.empty((superpixel.shape), dtype=np.uint8)
    all_maps = list()
    for label in range(5):
        (spx_indices,) = np.where(node_prediction == label)
        map_l = np.isin(superpixel - 1, spx_indices) * label
        all_maps.append(map_l)
    segmentation_map = np.stack(all_maps).sum(axis=0)
    return segmentation_map


def compute_segmentation_map_valentin(superpixel, node_prediction):
    segmentation_map = np.empty((superpixel.shape), dtype=np.uint8)
    translation = {k: v.item() for k, v in enumerate(node_label, 1)}
    fast_func = np.vectorize(lambda x: translation[x])
    segmentation_map = fast_func(superpixel)
    return segmentation_map

output_file = Path("downsampling_benchmark.csv")
if not output_file.exists():
    with open(output_file, "w") as file:
        file.write(
            "superpixels,downsampling,loading,baseline,segmentation_total,segmentation_additional,best_mean_iou,best_mean_iou_total,best_mean_iou_percentage\n"
        )
for RUN_ID in ["6b74c06337fb4feab67b556fe9e9efc3", "e32282480e0e4980b423e5aa53752b97"]:
    model = mlflow.pytorch.load_model(
        f"s3://mlflow/631/{RUN_ID}/artifacts/best.valid.node.NodeClassificationAccuracy",
        map_location=torch.device("cpu"),
    )
    run = mlflow.get_run(RUN_ID).to_dictionary()

    graph_directory = run["data"]["params"]["data.graph_directory"]
    validation_slides = list(
        map(int, run["data"]["params"]["data.validation_slides"][1:-1].split(","))
    )
    centroid_features = run["data"]["params"]["data.centroid_features"]

    superpixel_path = PREPROCESS_PATH / graph_directory / ".." / ".."

    all_metadata = merge_metadata(
        pd.read_pickle(IMAGES_DF),
        pd.read_pickle(ANNOTATIONS_DF),
        graph_directory=PREPROCESS_PATH / graph_directory,
        superpixel_directory=Path(superpixel_path),
        add_image_sizes=True,
    )

    validation_metadata = all_metadata[all_metadata.slide.isin(validation_slides)]

    annotations = GraphClassificationDataset(
        validation_metadata,
        patch_size=None,
        num_classes=NR_CLASSES,
        background_index=BACKGROUND_CLASS,
        centroid_features=centroid_features,
        return_segmentation_info=True,
        segmentation_downsample_ratio=1,
    ).annotations

    for DOWNSAMPLING in [1, 2, 4, 8, 16, 32]:

        print(f"NR_SUPERPIXELS: {graph_directory}")
        print(f"DOWNSAMPLING: {DOWNSAMPLING}")

        time_before_loading = datetime.datetime.now()
        validation_dataset = GraphClassificationDataset(
            validation_metadata,
            patch_size=None,
            num_classes=NR_CLASSES,
            background_index=BACKGROUND_CLASS,
            centroid_features=centroid_features,
            return_segmentation_info=True,
            segmentation_downsample_ratio=DOWNSAMPLING,
        )
        time_for_loading = (
            datetime.datetime.now() - time_before_loading
        ).total_seconds()
        print(f"Loading: {time_for_loading}")

        loader = DataLoader(
            validation_dataset, batch_size=4, collate_fn=collate, num_workers=6
        )
        loader.dataset.return_segmentation_info = False
        time_before_forward = datetime.datetime.now()
        for (
            graph,
            graph_label,
            node_label,
        ) in loader:
            graph_logits, node_logits = model(graph)
        time_for_forward = (
            datetime.datetime.now() - time_before_forward
        ).total_seconds()
        loader.dataset.return_segmentation_info = True
        print(f"Forward: {time_for_forward}")

        loader = DataLoader(
            validation_dataset, batch_size=4, collate_fn=collate_valid, num_workers=6
        )
        time_before_segmentation_masks = datetime.datetime.now()
        for graph, graph_label, node_label, annotation, superpixel in loader:
            graph_logits, node_logits = model(graph)
            batch_node_predictions = node_logits.argmax(axis=1)

            segmentation_maps1 = np.empty((superpixel.shape), dtype=np.uint8)

            start = 0
            for i, end in enumerate(graph.batch_num_nodes):
                node_predictions = batch_node_predictions[start : start + end].numpy()

                all_maps = list()
                for label in range(4):
                    (spx_indices,) = np.where(node_predictions == label)
                    map_l = np.isin(superpixel[i] - 1, spx_indices) * label
                    all_maps.append(map_l)
                segmentation_maps1[i] = np.stack(all_maps).sum(axis=0)

                start += end
        time_for_segmentation_masks = (
            datetime.datetime.now() - time_before_segmentation_masks
        ).total_seconds()
        print(f"Segmentation masks: {time_for_segmentation_masks}")

        mean_iou_metric = MeanIoU(nr_classes=4)
        mean_ious = list()

        for annotation, (graph, graph_label, node_label, _, superpixel) in zip(
            annotations, validation_dataset
        ):
            best_segmentation_map = compute_segmentation_map_guillaume(
                superpixel, node_label
            )
            best_segmentation_map = cv2.resize(
                best_segmentation_map,
                annotation.shape,
                interpolation=cv2.INTER_NEAREST,
            )
            mean_iou = mean_iou_metric(
                torch.Tensor(annotation), torch.Tensor(best_segmentation_map)
            )
            mean_ious.append(mean_iou)
        print(f"Additional: {time_for_segmentation_masks - time_for_forward}")
        print(f"Mean IoU: {np.mean(mean_ious):2f}")

        baseline = 1
        if DOWNSAMPLING == 1:
            baseline = np.mean(mean_ious)
            relative = 0.0
            percentage = 0.0
        else:
            relative = baseline - np.mean(mean_ious)
            percentage = (baseline / np.mean(mean_ious)) - 1
        print(f"Relative: {relative:2f} {percentage:2f}%")

        with open("downsampling_benchmark.csv", "a") as file:
            file.write(
                ",".join(
                    [
                        str(graph_directory),
                        str(DOWNSAMPLING),
                        str(time_for_loading),
                        str(time_for_forward),
                        str(time_for_segmentation_masks),
                        str(time_for_segmentation_masks - time_for_forward),
                        str(np.mean(mean_ious)),
                        str(relative),
                        str(percentage),
                    ]
                )
                + "\n"
            )
