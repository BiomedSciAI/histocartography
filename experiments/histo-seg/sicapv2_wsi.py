""" This module handles functions and constants related to the SiCAPv2 WSI dataset """

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset

from dataset import AugmentedGraphClassificationDataset
from logging_helper import log_preprocessing_parameters
from utils import merge_metadata

with os.popen("hostname") as subprocess:
    hostname = subprocess.read()
if hostname.startswith("zhcc"):
    BASE_PATH = Path("/dataP/anv/data/SICAPv2_WSI")
elif hostname.startswith("Gerbil"):
    BASE_PATH = Path("/Users/anv/Documents/SICAPv2_WSI")
else:
    assert False, f"Hostname {hostname} does not have the data path specified"

MASK_VALUE_TO_TEXT = {
    0: "Benign",
    1: "Gleason_3",
    2: "Gleason_4",
    3: "Gleason_5",
    4: "unlabelled",
}
MASK_VALUE_TO_COLOR = {0: "green", 1: "blue", 2: "yellow", 3: "red", 4: "white"}
NR_CLASSES = 4
BACKGROUND_CLASS = 4
VARIABLE_SIZE = True
ADDITIONAL_ANNOTATION = False
DISCARD_THRESHOLD = 5000
THRESHOLD = 0.003
WSI_FIX = True

IMAGE_PATH = BASE_PATH / "wsi"
ANNOTATION_PATH = BASE_PATH / "wsi_masks"
TRAIN_PARTIAL_ANNOTATION_PATHS = {
    50: BASE_PATH / "wsi_masks_partial" / "wsi_masks_50",
    25: BASE_PATH / "wsi_masks_partial" / "wsi_masks_25"
}
LABELS_PATH = BASE_PATH / "wsi_labels.xlsx"
PREPROCESS_PATH = BASE_PATH / "preprocess"
OUTPUT_PATH = PREPROCESS_PATH / "outputs"
IMAGES_DF = BASE_PATH / "images.pickle"
ANNOTATIONS_DF = BASE_PATH / "annotations.pickle"
PARTIAL_ANNOTATIONS_DFS = {
    50: BASE_PATH / "annotations50.pickle",
    25: BASE_PATH / "annotations25.pickle"
}
LABELS_DF = BASE_PATH / "image_level_annotations.pickle"

TEST_SPLIT_PATH = BASE_PATH / "partition" / "Test" / "Test.csv"
VALID_FOLDS = [1, 2, 3, 4]
TRAIN_SPLIT_PATHS = [
    BASE_PATH / "partition" / "Validation" / f"Val{x}" / "Train.csv"
    for x in VALID_FOLDS
]
VALID_SPLIT_PATHS = [
    BASE_PATH / "partition" / "Validation" / f"Val{x}" / "Test.csv" for x in VALID_FOLDS
]


def generate_images_meta_df():
    image_df = list()
    for path in IMAGE_PATH.iterdir():
        name = path.name.split(".")[0]
        image_df.append((name, path))
    image_df = pd.DataFrame(image_df, columns=["name", "path"]).set_index("name")
    image_df.to_pickle(IMAGES_DF)


def generate_annotations_meta_df():
    anntation_df = list()
    for path in ANNOTATION_PATH.iterdir():
        name = path.name.split(".")[0]
        anntation_df.append((name, path))
    anntation_df = pd.DataFrame(anntation_df, columns=["name", "path"]).set_index(
        "name"
    )
    anntation_df.to_pickle(ANNOTATIONS_DF)


def generate_partial_annotations_meta_df():
    for key, value in TRAIN_PARTIAL_ANNOTATION_PATHS.items():
        anntation_df = list()
        for path in value.iterdir():
            name = path.name.split(".")[0]
            anntation_df.append((name, path))
        anntation_df = pd.DataFrame(anntation_df, columns=["name", "path"]).set_index(
            "name"
        )
        anntation_df.to_pickle(PARTIAL_ANNOTATIONS_DFS[key])


def generate_labels():
    df = pd.read_excel(LABELS_PATH, engine="openpyxl")

    def score_to_label(x):
        if x == 0:
            return x
        else:
            return x - 2

    def primary_secondary_to_onehot(df):
        df = df.copy()
        df["Gleason_primary"] = df["Gleason_primary"].apply(score_to_label)
        df["Gleason_secondary"] = df["Gleason_secondary"].apply(score_to_label)
        new_df = list()
        for _, row in df.iterrows():
            b, g3, g4, g5 = to_onehot_with_ignore(
                np.unique([row["Gleason_primary"], row["Gleason_secondary"]])
            ).tolist()
            new_df.append((row["slide_id"], row["patient_id"], b, g3, g4, g5))
        return pd.DataFrame(
            new_df,
            columns=[
                "name",
                "patient_id",
                "benign",
                "grade3",
                "grade4",
                "grade5",
            ],
        ).set_index("name")

    def to_onehot_with_ignore(input_vector) -> torch.Tensor:
        input_vector = torch.Tensor(input_vector.astype(np.int64)).to(torch.int64)
        one_hot_vector = torch.nn.functional.one_hot(
            input_vector, num_classes=NR_CLASSES + 1
        )
        clean_one_hot_vector = torch.cat(
            [
                one_hot_vector[:, 0:BACKGROUND_CLASS],
                one_hot_vector[:, BACKGROUND_CLASS + 1 :],
            ],
            dim=1,
        )
        return clean_one_hot_vector.sum(dim=0)

    df = primary_secondary_to_onehot(df)
    df.to_pickle(LABELS_DF)


def to_mapper(df):
    mapper = dict()
    for name, row in df.iterrows():
        mapper[name] = np.array(
            [row["benign"], row["grade3"], row["grade4"], row["grade5"]]
        )
    return mapper


def get_metadata(graph_directory, partial_annotation=None):
    graph_directory = OUTPUT_PATH / graph_directory
    superpixel_directory = graph_directory / "superpixels"
    tissue_mask_directory = graph_directory / "tissue_masks"
    log_preprocessing_parameters(graph_directory)
    image_metadata = pd.read_pickle(IMAGES_DF)
    if partial_annotation is None:
        annotation_metadata = pd.read_pickle(ANNOTATIONS_DF)
    else:
        assert partial_annotation in PARTIAL_ANNOTATIONS_DFS, f"Partial Annotation {partial_annotation} not supported. Only {PARTIAL_ANNOTATIONS_DFS.keys()}"
        annotation_metadata = pd.read_pickle(PARTIAL_ANNOTATIONS_DFS[partial_annotation])
    all_metadata = merge_metadata(
        image_metadata=image_metadata,
        annotation_metadata=annotation_metadata,
        graph_directory=graph_directory,
        superpixel_directory=superpixel_directory,
        tissue_mask_directory=tissue_mask_directory,
        add_image_sizes=True,
    )
    labels_metadata = pd.read_pickle(LABELS_DF)
    label_mapper = to_mapper(labels_metadata)
    return all_metadata, label_mapper


def prepare_graph_datasets(
    graph_directory: str,
    fold: int,
    overfit_test: bool = False,
    centroid_features: str = "no",
    downsample_segmentation_maps: int = 1,
    augmentation_mode: Optional[str] = None,
    supervision: Optional[dict] = None,
    patch_size: Optional[int] = None,
    partial_annotation: Optional[int] = None,
    additional_training_arguments: dict = {},
    additional_validation_arguments: dict = {},
) -> Tuple[Dataset, Dataset]:
    assert fold in VALID_FOLDS, f"Fold must be in {VALID_FOLDS} but is {fold}"
    all_metadata, label_mapper = get_metadata(graph_directory, partial_annotation)
    training_names = pd.read_csv(TRAIN_SPLIT_PATHS[fold-1], index_col=0)[
        "wsi_name"
    ].values
    validation_names = pd.read_csv(VALID_SPLIT_PATHS[fold-1], index_col=0)[
        "wsi_name"
    ].values
    assert set(training_names).isdisjoint(
        set(validation_names)
    ), f"Train and valid are not disjoint for fold {fold}"
    training_metadata = all_metadata.loc[training_names]
    validation_metadata = all_metadata.loc[validation_names]

    if overfit_test:
        training_metadata = training_metadata.sample(2)
        validation_metadata = validation_metadata.sample(2)

    if patch_size is None:
        patch_size_augmentation = None
    else:
        patch_size_augmentation = (patch_size, patch_size)

    training_arguments = {
        "num_classes": NR_CLASSES,
        "background_index": BACKGROUND_CLASS,
        "centroid_features": centroid_features,
        "image_label_mapper": label_mapper,
        "patch_size": patch_size_augmentation
    }
    training_arguments.update(additional_training_arguments)
    validation_arguments = {
        "num_classes": NR_CLASSES,
        "background_index": BACKGROUND_CLASS,
        "centroid_features": centroid_features,
        "return_segmentation_info": True,
        "segmentation_downsample_ratio": downsample_segmentation_maps,
        "image_label_mapper": label_mapper,
    }
    validation_arguments.update(additional_validation_arguments)

    supervision_mode = (
        supervision.get("mode", "tissue_level")
        if supervision is not None
        else "tissue_level"
    )
    if supervision_mode == "tissue_level":
        training_arguments["tissue_metadata"] = training_metadata
        training_arguments["image_metadata"] = None
        validation_arguments["tissue_metadata"] = validation_metadata
        validation_arguments["image_metadata"] = None
    elif supervision_mode == "image_level":
        training_arguments["tissue_metadata"] = None
        training_arguments["image_metadata"] = training_metadata
        validation_arguments["tissue_metadata"] = validation_metadata
        validation_arguments["image_metadata"] = None
    else:
        raise NotImplementedError

    training_dataset = AugmentedGraphClassificationDataset(
        augmentation_mode=augmentation_mode, **training_arguments
    )
    validation_dataset = AugmentedGraphClassificationDataset(
        augmentation_mode=None, **validation_arguments
    )
    return training_dataset, validation_dataset


def prepare_graph_testset(
    graph_directory: str,
    test: bool = False,
    centroid_features: str = "no",
    **kwargs,
) -> Dataset:
    all_metadata, label_mapper = get_metadata(graph_directory)
    test_names = pd.read_csv(TEST_SPLIT_PATH, index_col=0)["wsi_name"].values
    test_metadata = all_metadata.loc[test_names]

    if test:
        test_metadata = test_metadata.sample(1)

    test_arguments = {
        "num_classes": NR_CLASSES,
        "background_index": BACKGROUND_CLASS,
        "centroid_features": centroid_features,
        "return_segmentation_info": True,
        "image_label_mapper": label_mapper,
        "augmentation_mode": None,
    }
    test_arguments.update(kwargs)
    test_dataset = AugmentedGraphClassificationDataset(
        tissue_metadata=test_metadata, **test_arguments
    )
    return test_dataset


def show_class_acivation(per_class_output):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(per_class_output[i], vmin=0, vmax=1, cmap="viridis")
        ax.set_axis_off()
        ax.set_title(MASK_VALUE_TO_TEXT[i])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig


def show_segmentation_masks(output, annotation=None, **kwargs):
    height = 4
    width = 5
    ncols = 1
    if annotation is not None:
        width += 5
        ncols += 1
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(width, height))
    cmap = ListedColormap(MASK_VALUE_TO_COLOR.values())

    mask_ax = ax if annotation is None else ax[0]
    im = mask_ax.imshow(output, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
    mask_ax.axis("off")
    if annotation is not None:
        ax[1].imshow(
            annotation, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest"
        )
        ax[1].axis("off")
        ax[0].set_title("Prediction")
        ax[1].set_title("Ground Truth")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.016, 0.7])
    cbar = fig.colorbar(im, ticks=[0, 1, 2, 3, 4], cax=cbar_ax)
    cbar.ax.set_yticklabels(MASK_VALUE_TO_TEXT.values())
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["cleanup", "labels", "partial_annotations"], default="cleanup")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command == "cleanup":
        generate_images_meta_df()
        generate_annotations_meta_df()
        generate_labels()
    elif args.command == "partial_annotations":
        generate_partial_annotations_meta_df()
