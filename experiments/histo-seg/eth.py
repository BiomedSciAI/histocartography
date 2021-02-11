"""This module handles functions and constants related to the ETH dataset"""

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from histocartography.preprocessing.utils import fast_histogram
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm

from dataset import (
    AugmentedGraphClassificationDataset,
    GraphClassificationDataset,
    ImageDataset,
    PatchClassificationDataset,
)
from logging_helper import log_preprocessing_parameters
from utils import find_superpath, merge_metadata, read_image

with os.popen("hostname") as subprocess:
    hostname = subprocess.read()
if hostname.startswith("zhcc"):
    BASE_PATH = Path("/dataP/anv/data/ETH")
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
TRAIN_IMAGE_LEVEL_ANNOTATIONS = BASE_PATH / "image_level_train.csv"
TEST_IMAGE_LEVEL_ANNOTATIONS = BASE_PATH / "image_level_test.csv"

PREPROCESS_PATH = BASE_PATH / "preprocess"
DATASET_PATH = BASE_PATH / "datasets"

IMAGES_DF = BASE_PATH / "images.pickle"
ANNOTATIONS_DF = BASE_PATH / "annotations.pickle"
LABELS_DF = BASE_PATH / "image_level_annotations.pickle"

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
    for images_folder in tqdm(
        list(filter(lambda x: x.name.startswith(IMAGE_PREFIX), BASE_PATH.iterdir())),
        desc="Copy images",
    ):
        for image_path in images_folder.iterdir():
            shutil.copy(image_path, TMA_IMAGE_PATH)
        if delete:
            shutil.rmtree(images_folder)


def generate_images_meta_df() -> None:
    """Generate the images metadata data frame and save it to IMAGES_DF"""
    df = list()
    for image_path in tqdm(TMA_IMAGE_PATH.iterdir(), desc="DF images"):
        # Handle OSX madness
        if image_path.name == ".DS_Store":
            continue
        if image_path.is_dir():
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

    for pathologist, new_pathologist_path in TEST_ANNOTATIONS_PATHS:
        old_pathologist_path = BASE_PATH / new_pathologist_path.name
        if not new_pathologist_path.exists():
            new_pathologist_path.mkdir()
        for file in tqdm(
            old_pathologist_path.iterdir(), f"Masks pathologist {pathologist}"
        ):
            shutil.copy(file, new_pathologist_path)
        if delete:
            shutil.rmtree(old_pathologist_path)


def generate_annotations_meta_df() -> None:
    """Generate the annotations metadata data frame and save it to ANNOTATIONS_DF"""
    df = list()
    for file in (TRAIN_ANNOTATION_PATH).iterdir():
        if file.name == ".DS_Store":
            continue
        if file.is_dir():
            continue
        name = file.name.split(".")[0].split(MASK_PREFIX)[1]
        df.append((name, file, 1, "train"))

    for pathologist, pathologist_path in TEST_ANNOTATIONS_PATHS:
        for file in tqdm(
            pathologist_path.iterdir(), desc=f"DF Masks pathologist {pathologist}"
        ):
            if file.name == ".DS_Store":
                continue
            if file.is_dir():
                continue
            name = file.name.split(".")[0].split(
                MASK_PREFIX[:-1] + str(pathologist) + MASK_PREFIX[-1]
            )[1]
            df.append((name, file, pathologist, "test"))
    df = pd.DataFrame(df, columns=["name", "path", "pathologist", "use"])
    df.to_pickle(ANNOTATIONS_DF)


def generate_labels() -> None:
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

    def gleason_grade(labels):
        b, g3, g4, g5 = labels
        if g3 + g4 + g5 == 0:
            return "benign"
        else:
            if g3 == 1 and g4 == 0 and g5 == 0:
                return "grade6"
            elif g3 == 1 and g4 == 1 and g5 == 0:
                return "grade7"
            elif g3 == 0 and g4 == 1 and g5 == 0:
                return "grade8"
            elif g3 == 0 and g4 == 1 and g5 == 1:
                return "grade9"
            elif g3 == 0 and g4 == 0 and g5 == 1:
                return "grade10"
            elif g5 == 1 and g4 == 1:
                return "grade9"
            elif g5 == 1 and g4 == 0 and g3 == 1:
                print(f"Weird grade: {labels}")
                return "grade8"
            else:
                print(f"Weird grade: {labels}")
                return "grade7"

    def read_image_level_csv(path):
        df = pd.read_csv(path)
        df = df.replace(to_replace=np.nan, value=0.0)
        for column in ["benign", "grade3", "grade4", "grade5"]:
            df[column] = df[column].apply(int)
        return df

    def read_image_level_annotations(
        train=TRAIN_IMAGE_LEVEL_ANNOTATIONS, test=TEST_IMAGE_LEVEL_ANNOTATIONS
    ):
        train_df = read_image_level_csv(train)
        train_df["use"] = "train"
        test_df = read_image_level_csv(test)
        test_df["use"] = "test"
        df = train_df.append(test_df, ignore_index=True)
        df["pathologist"] = 3
        df = df.rename(columns={"image_name": "name"})
        return df

    def add_gleason_score(df):
        grades = list()
        for _, row in df.iterrows():
            grades.append(
                gleason_grade(
                    (row["benign"], row["grade3"], row["grade4"], row["grade5"])
                )
            )
        df["gleason_grade"] = grades
        return df

    image_df = pd.read_pickle(ANNOTATIONS_DF)
    image_level_df = list()
    for _, row in tqdm(image_df.iterrows(), total=len(image_df)):
        annotation = read_image(row["path"])
        labels = np.where(fast_histogram(annotation, NR_CLASSES))[0]
        labels = to_onehot_with_ignore(labels)
        labels = tuple(labels.tolist())
        image_level_df.append((row["name"], row["pathologist"], row["use"]) + labels)
    image_level_df = pd.DataFrame(
        image_level_df,
        columns=["name", "pathologist", "use", "benign", "grade3", "grade4", "grade5"],
    )
    original_df = add_gleason_score(image_level_df)

    additional_df = read_image_level_annotations()
    additional_df = add_gleason_score(additional_df)

    combined_df = original_df.append(additional_df, ignore_index=True)
    combined_df.to_pickle(BASE_PATH / "image_level_annotations.pickle")


def select_label(df, mode="new_labels"):
    all_names = pd.unique(df.name)
    if mode == "new_labels":
        new_df = df[df["pathologist"] == 3].set_index("name")
        old_df = df[df["pathologist"] == 1].set_index("name")
        new_names = pd.unique(new_df.index)
        missing_names = set(all_names) - set(new_names)
        missing_df = old_df.loc[missing_names]
        df = new_df.copy()
        df = df.append(missing_df)
    elif mode == "original_labels":
        df = df[df["pathologist"] == 1].set_index("name").copy()
    return df


def to_mapper(df):
    mapper = dict()
    for name, row in df.iterrows():
        mapper[name] = np.array(
            [row["benign"], row["grade3"], row["grade4"], row["grade5"]]
        )
    return mapper


def prepare_graph_datasets(
    graph_directory: str,
    patch_size: Optional[int] = None,
    use_patches_for_validation: bool = False,
    training_slides: Optional[List[int]] = (111, 199, 204),
    validation_slides: Optional[List[int]] = (76,),
    train_fraction: Optional[float] = None,
    overfit_test: bool = False,
    centroid_features: str = "no",
    normalize_features: bool = False,
    downsample_segmentation_maps: int = 1,
    tissue_mask_directory: Optional[str] = None,
    use_augmentation_dataset: bool = False,
    augmentation_mode: Optional[bool] = False,
    image_labels_mode: Optional[str] = "original_labels",
    supervision: Optional[dict] = None,
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
    assert train_fraction is not None or (
        training_slides is not None and validation_slides is not None
    )

    graph_directory = PREPROCESS_PATH / graph_directory
    superpixel_directory = graph_directory / "superpixels"
    if not superpixel_directory.exists():
        superpixel_directory = graph_directory / ".." / ".."
    new_tissue_mask_directory = graph_directory / "tissue_masks"
    if not new_tissue_mask_directory.exists():
        new_tissue_mask_directory = find_superpath(
            graph_directory, tissue_mask_directory
        )
    log_preprocessing_parameters(graph_directory)
    all_metadata = merge_metadata(
        pd.read_pickle(IMAGES_DF),
        pd.read_pickle(ANNOTATIONS_DF),
        graph_directory=graph_directory,
        superpixel_directory=superpixel_directory,
        tissue_mask_directory=new_tissue_mask_directory,
        add_image_sizes=True,
    )
    labels_metadata = pd.read_pickle(LABELS_DF)
    training_label_mapper = to_mapper(
        select_label(labels_metadata, mode=image_labels_mode)
    )
    validation_label_mapper = to_mapper(
        select_label(labels_metadata, mode="original_labels")
    )
    if train_fraction is not None:
        train_indices, validation_indices = train_test_split(
            all_metadata.index.values, train_size=train_fraction
        )
        training_metadata = all_metadata.loc[train_indices]
        validation_metadata = all_metadata.loc[validation_indices]
    else:
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
        training_metadata = training_metadata.sample(38)
        validation_metadata = validation_metadata.sample(10)

    if patch_size is None:
        patch_size_augmentation = None
    else:
        patch_size_augmentation = (patch_size, patch_size)

    training_arguments = {
        "patch_size": patch_size_augmentation,
        "num_classes": NR_CLASSES,
        "background_index": BACKGROUND_CLASS,
        "centroid_features": centroid_features,
        "mean": precomputed_mean,
        "std": precomputed_std,
        "image_label_mapper": training_label_mapper,
    }
    validation_arguments = {
        "patch_size": patch_size_augmentation if use_patches_for_validation else None,
        "num_classes": NR_CLASSES,
        "background_index": BACKGROUND_CLASS,
        "centroid_features": centroid_features,
        "mean": precomputed_mean,
        "std": precomputed_std,
        "return_segmentation_info": True,
        "segmentation_downsample_ratio": downsample_segmentation_maps,
        "image_label_mapper": validation_label_mapper,
    }

    # Handle supervision modes
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

    if use_augmentation_dataset:
        training_dataset = AugmentedGraphClassificationDataset(
            augmentation_mode=augmentation_mode, **training_arguments
        )
        validation_dataset = AugmentedGraphClassificationDataset(
            augmentation_mode=None, **validation_arguments
        )
    else:
        training_dataset = GraphClassificationDataset(**training_arguments)
        validation_dataset = GraphClassificationDataset(**validation_arguments)

    return training_dataset, validation_dataset


def prepare_graph_testset(
    graph_directory: str,
    test: bool = False,
    centroid_features: str = "no",
    normalize_features: bool = False,
    test_slides: Optional[List[int]] = TEST_SLIDES,
    use_augmentation_dataset: bool = False,
    augmentation_mode: Optional[bool] = False,
    image_labels_mode: Optional[str] = "original_labels",
    **kwargs,
) -> Dataset:
    graph_directory = PREPROCESS_PATH / graph_directory

    annotation_metadata = pd.read_pickle(ANNOTATIONS_DF)
    pathologist2_metadata = annotation_metadata[
        (annotation_metadata.use == "test") & (annotation_metadata.pathologist == 2)
    ]
    pathologist2_metadata = pathologist2_metadata.set_index("name")
    pathologist2_metadata = pathologist2_metadata.rename(
        columns={"path": "annotation2_path"}
    )
    log_preprocessing_parameters(graph_directory)
    superpixel_directory = graph_directory / "superpixels"
    tissue_mask_directory = graph_directory / "tissue_masks"
    all_metadata = merge_metadata(
        pd.read_pickle(IMAGES_DF),
        pd.read_pickle(ANNOTATIONS_DF),
        graph_directory=graph_directory,
        superpixel_directory=superpixel_directory,
        tissue_mask_directory=tissue_mask_directory,
        add_image_sizes=True,
    )
    labels_metadata = pd.read_pickle(LABELS_DF)
    label_mapper = to_mapper(select_label(labels_metadata, mode="original_labels"))
    test_metadata = all_metadata[all_metadata.slide.isin(test_slides)]
    test_metadata = test_metadata.join(pathologist2_metadata[["annotation2_path"]])

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

    if test:
        test_metadata = test_metadata.sample(5)

    test_arguments = {
        "patch_size": None,
        "num_classes": NR_CLASSES,
        "background_index": BACKGROUND_CLASS,
        "centroid_features": centroid_features,
        "mean": precomputed_mean,
        "std": precomputed_std,
        "return_segmentation_info": True,
        "image_label_mapper": label_mapper,
    }
    test_arguments.update(kwargs)
    if use_augmentation_dataset:
        test_dataset = AugmentedGraphClassificationDataset(
            tissue_metadata=test_metadata, augmentation_mode=None, **test_arguments
        )
    else:
        test_dataset = GraphClassificationDataset(
            tissue_metadata=test_metadata, **test_arguments
        )
    return test_dataset


def prepare_patch_datasets(
    image_path: str,
    tissue_mask_directory: str,
    training_slides: Optional[List[int]] = None,
    validation_slides: Optional[List[int]] = None,
    train_fraction: Optional[float] = None,
    overfit_test: bool = False,
    normalizer: Optional[dict] = None,
    drop_multiclass_patches: bool = False,
    drop_unlabelled_patches: bool = False,
    drop_tissue_patches: float = 0.0,
    drop_validation_patches: bool = False,
    **kwargs,
) -> Tuple[Dataset, Dataset]:

    image_path = PREPROCESS_PATH / image_path
    all_metadata = merge_metadata(
        pd.read_pickle(IMAGES_DF),
        pd.read_pickle(ANNOTATIONS_DF),
        processed_image_directory=image_path,
        add_image_sizes=True,
        tissue_mask_directory=image_path / tissue_mask_directory,
    )

    if train_fraction is not None:
        train_indices, validation_indices = train_test_split(
            all_metadata.index.values, train_size=train_fraction
        )
        training_metadata = all_metadata.loc[train_indices]
        validation_metadata = all_metadata.loc[validation_indices]
    else:
        training_metadata = all_metadata[all_metadata.slide.isin(training_slides)]
        validation_metadata = all_metadata[all_metadata.slide.isin(validation_slides)]

    if overfit_test:
        training_metadata = training_metadata.sample(1)
        validation_metadata = validation_metadata.sample(1)

    if normalizer is not None:
        mean = torch.Tensor(normalizer["mean"])
        std = torch.Tensor(normalizer["std"])
    else:
        mean = torch.Tensor([0, 0, 0])
        std = torch.Tensor([1, 1, 1])

    augmentations = kwargs.pop("augmentations")

    training_dataset = PatchClassificationDataset(
        metadata=training_metadata,
        num_classes=NR_CLASSES,
        background_index=BACKGROUND_CLASS,
        mean=mean,
        std=std,
        augmentations=augmentations,
        **kwargs,
    )
    validation_dataset = PatchClassificationDataset(
        metadata=validation_metadata,
        num_classes=NR_CLASSES,
        background_index=BACKGROUND_CLASS,
        mean=mean,
        std=std,
        **kwargs,
    )

    if drop_multiclass_patches:
        training_dataset.drop_confusing_patches()
    elif drop_unlabelled_patches:
        training_dataset.drop_unlablled_patches()
    if drop_tissue_patches > 0.0:
        training_dataset.drop_tissueless_patches(minimum_fraction=drop_tissue_patches)
    if drop_validation_patches:
        if drop_multiclass_patches:
            validation_dataset.drop_confusing_patches()
        if drop_tissue_patches > 0.0:
            validation_dataset.drop_tissueless_patches(
                minimum_fraction=drop_tissue_patches
            )

    return training_dataset, validation_dataset


def prepare_patch_testset(
    image_path: str,
    test: bool,
    tissue_mask_directory: Optional[str] = None,
    test_slides: Optional[List[int]] = TEST_SLIDES,
    normalizer: Optional[dict] = None,
    **kwargs,
) -> Dataset:

    annotation_metadata = pd.read_pickle(ANNOTATIONS_DF)
    pathologist2_metadata = annotation_metadata[
        (annotation_metadata.use == "test") & (annotation_metadata.pathologist == 2)
    ]
    pathologist2_metadata = pathologist2_metadata.set_index("name")
    pathologist2_metadata = pathologist2_metadata.rename(
        columns={"path": "annotation2_path"}
    )

    all_metadata = merge_metadata(
        pd.read_pickle(IMAGES_DF),
        annotation_metadata,
        processed_image_directory=PREPROCESS_PATH / image_path,
        tissue_mask_directory=PREPROCESS_PATH / image_path / tissue_mask_directory
        if tissue_mask_directory is not None
        else None,
        add_image_sizes=True,
    )
    test_metadata = all_metadata[all_metadata.slide.isin(test_slides)]
    test_metadata = test_metadata.join(pathologist2_metadata[["annotation2_path"]])

    if test:
        test_metadata = test_metadata.sample(1)

    if normalizer is not None:
        mean = normalizer["mean"]
        std = normalizer["std"]
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]

    return ImageDataset(
        metadata=test_metadata,
        num_classes=NR_CLASSES,
        background_index=BACKGROUND_CLASS,
        mean=mean,
        std=std,
        **kwargs,
    )


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


def show_segmentation_masks(output, annotation=None, annotation2=None):
    height = 4
    width = 5
    ncols = 1
    if annotation is not None:
        width += 5
        ncols += 1
        if annotation2 is not None:
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
        if annotation2 is not None:
            ax[2].imshow(
                annotation2, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest"
            )
            ax[2].axis("off")
            ax[1].set_title("Ground Truth 1")
            ax[2].set_title("Ground Truth 2")
        else:
            ax[1].set_title("Ground Truth")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.016 if annotation2 is None else 0.01, 0.7])
    cbar = fig.colorbar(im, ticks=[0, 1, 2, 3, 4], cax=cbar_ax)
    cbar.ax.set_yticklabels(MASK_VALUE_TO_TEXT.values())
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["cleanup", "labels"], default="cleanup")
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
    elif args.command == "labels":
        generate_labels()
