"""This module handles functions and constants related to the ETH dataset"""

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

from dataset import GraphClassificationDataset
from utils import merge_metadata

with os.popen("hostname") as subprocess:
    hostname = subprocess.read()
if hostname.startswith("zhcc"):
    BASE_PATH = Path("/dataT/anv/Data/ETH")
elif hostname.startswith("Gerbil"):
    BASE_PATH = Path("/Users/anv/Documents/ETH")
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

TMA_IMAGE_PATH = BASE_PATH / "TMA_Images"
TRAIN_ANNOTATION_PATH = BASE_PATH / "Gleason_masks_train"
TEST_ANNOTATION_PATH = BASE_PATH / "Gleason_masks_test"
TEST_ANNOTATION = "Gleason_masks_test_pathologist"
TEST_PATHOLOGISTS = [1, 2]
TEST_ANNOTATIONS_PATHS = [
    (i, TEST_ANNOTATION_PATH / f"{TEST_ANNOTATION}{i}") for i in TEST_PATHOLOGISTS
]

PREPROCESS_PATH = BASE_PATH / "preprocess"
DATASET_PATH = BASE_PATH / "datasets"

IMAGES_DF = BASE_PATH / "images.pickle"
ANNOTATIONS_DF = BASE_PATH / "annotations.pickle"

MASK_PREFIX = "mask_"
IMAGE_PREFIX = "ZT"

TRAINING_SLIDES = [111, 199, 204, 76]
TEST_SLIDES = [80]


def generate_image_folder_structure(delete: bool = False) -> None:
    """Convert the download structure of images into the desired structure

    Args:
        delete (bool, optional): Whether to delete the old files. Defaults to False.
    """
    # Copy images
    if not TMA_IMAGE_PATH.exists():
        TMA_IMAGE_PATH.mkdir()
    for images_folder in filter(
        lambda x: x.name.startswith(IMAGE_PREFIX), BASE_PATH.iterdir()
    ):
        for image_path in images_folder.iterdir():
            shutil.copy(image_path, TMA_IMAGE_PATH)
        if delete:
            shutil.rmtree(images_folder)


def generate_images_meta_df() -> None:
    """Generate the images metadata data frame and save it to IMAGES_DF"""
    df = list()
    for image_path in TMA_IMAGE_PATH.iterdir():
        # Handle OSX madness
        if image_path.name == ".DS_Store":
            continue
        slide = image_path.name.split("_")[0]
        slide = int(slide[len(IMAGE_PREFIX) :])
        if slide in TRAINING_SLIDES:
            target_folder = "train"
        elif slide in TEST_SLIDES:
            target_folder = "test"
        else:
            print(f"Slide {slide} ignored")
            continue
        name = image_path.name.split(".")[0]
        df.append((name, image_path, target_folder, slide))
    df = pd.DataFrame(df, columns=["name", "path", "use", "slide"])
    df = df.set_index("name")
    df.to_pickle(IMAGES_DF)


def generate_mask_folder_structure(delete: bool = False) -> None:
    """Convert the download structure of masks into the desired structure

    Args:
        delete (bool, optional): Whether to delete the old files. Defaults to False.
    """
    downloaded_train_path = BASE_PATH / "Gleason_masks_train"

    if not TRAIN_ANNOTATION_PATH.exists():
        TRAIN_ANNOTATION_PATH.mkdir()

    for file in (BASE_PATH / downloaded_train_path).iterdir():
        if TRAIN_ANNOTATION_PATH != downloaded_train_path:
            shutil.copy(file, TRAIN_ANNOTATION_PATH)
    if TRAIN_ANNOTATION_PATH != downloaded_train_path and delete:
        shutil.rmtree(downloaded_train_path)

    if not TEST_ANNOTATION_PATH.exists():
        TEST_ANNOTATION_PATH.mkdir()

    for _, new_pathologist_path in TEST_ANNOTATIONS_PATHS:
        old_pathologist_path = BASE_PATH / new_pathologist_path.name
        if not new_pathologist_path.exists():
            new_pathologist_path.mkdir()
        for file in old_pathologist_path.iterdir():
            shutil.copy(file, new_pathologist_path)
        if delete:
            shutil.rmtree(old_pathologist_path)


def generate_annotations_meta_df() -> None:
    """Generate the annotations metadata data frame and save it to ANNOTATIONS_DF"""
    df = list()
    for file in (TRAIN_ANNOTATION_PATH).iterdir():
        if file.name == ".DS_Store":
            continue
        name = file.name.split(".")[0].split(MASK_PREFIX)[1]
        df.append((name, file, 1, "train"))

    for pathologist, pathologist_path in TEST_ANNOTATIONS_PATHS:
        for file in pathologist_path.iterdir():
            if file.name == ".DS_Store":
                continue
            name = file.name.split(".")[0].split(
                MASK_PREFIX[:-1] + str(pathologist) + MASK_PREFIX[-1]
            )[1]
            df.append((name, file, pathologist, "test"))
    df = pd.DataFrame(df, columns=["name", "path", "pathologist", "use"])
    df.to_pickle(ANNOTATIONS_DF)


def prepare_datasets(
    graph_directory: str,
    training_slides: List[int],
    validation_slides: List[int],
    patch_size: int,
    use_patches_for_validation: bool,
    overfit_test: bool = False,
    centroid_features: str = "no",
    normalize_features: bool = False,
) -> Tuple[Dataset, Dataset]:
    """Create the datset from the hardcoded values in this file as well as dynamic information

    Args:
        graph_directory (Path): Directory of the dumped graphs
        training_slides (List[int]): List of slides to use for training
        validation_slides (List[int]): List of slides to use for validation
        patch_size (int): Size of the patches

    Returns:
        Tuple[Dataset, Dataset]: Training set, validation set
    """
    graph_directory = PREPROCESS_PATH / graph_directory
    all_metadata = merge_metadata(
        pd.read_pickle(IMAGES_DF),
        pd.read_pickle(ANNOTATIONS_DF),
        graph_directory=graph_directory,
        add_image_sizes=True,
    )
    training_metadata = all_metadata[all_metadata.slide.isin(training_slides)]
    validation_metadata = all_metadata[all_metadata.slide.isin(validation_slides)]

    if normalize_features:
        precomputed_mean = torch.load(
            PREPROCESS_PATH
            / "outputs"
            / "normalizers"
            / f"mean_{graph_directory.name}.pth"
        )
        precomputed_std = torch.load(
            PREPROCESS_PATH
            / "outputs"
            / "normalizers"
            / f"std_{graph_directory.name}.pth"
        )
    else:
        precomputed_mean = None
        precomputed_std = None

    if overfit_test:
        training_metadata = training_metadata.sample(1)
        validation_metadata = validation_metadata.sample(1)

    training_dataset = GraphClassificationDataset(
        training_metadata,
        patch_size=(patch_size, patch_size),
        num_classes=NR_CLASSES,
        background_index=BACKGROUND_CLASS,
        centroid_features=centroid_features,
        mean=precomputed_mean,
        std=precomputed_std,
    )
    validation_dataset = GraphClassificationDataset(
        validation_metadata,
        patch_size=(patch_size, patch_size) if use_patches_for_validation else None,
        num_classes=NR_CLASSES,
        background_index=BACKGROUND_CLASS,
        centroid_features=centroid_features,
        mean=precomputed_mean,
        std=precomputed_std,
    )

    return training_dataset, validation_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["cleanup"], default="cleanup")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command == "cleanup":
        generate_image_folder_structure(delete=True)
        generate_images_meta_df()
        generate_mask_folder_structure(delete=True)
        generate_annotations_meta_df()
