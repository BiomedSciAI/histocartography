"""This module handles functions and constants related to the ETH dataset"""

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

BASE_PATH = Path("/Users/anv/Documents/ETH")

MASK_VALUE_TO_TEXT = {
    0: "Benign",
    1: "Gleason_3",
    2: "Gleason_4",
    3: "Gleason_5",
    4: "unlabelled",
}

MASK_VALUE_TO_COLOR = {0: "green", 1: "blue", 2: "yellow", 3: "red", 4: "white"}

TMA_IMAGE_PATH = BASE_PATH / "TMA_Images"
TRAIN_ANNOTATION_PATH = BASE_PATH / "Gleason_masks_train"
TEST_ANNOTATION_PATH = BASE_PATH / "Gleason_masks_test"
TEST_ANNOTATION = "Gleason_masks_test_pathologist"
TEST_PATHOLOGISTS = [1, 2]
TEST_ANNOTATIONS_PATHS = [
    (i, TEST_ANNOTATION_PATH / f"{TEST_ANNOTATION}{i}") for i in TEST_PATHOLOGISTS
]

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

    for _, pathologist_path in TEST_ANNOTATIONS_PATHS:
        new_pathologist_path = TEST_ANNOTATION_PATH / pathologist_path.name
        if not new_pathologist_path.exists():
            new_pathologist_path.mkdir()
        for file in pathologist_path.iterdir():
            shutil.copy(file, new_pathologist_path)
        if delete:
            shutil.rmtree(pathologist_path)


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
