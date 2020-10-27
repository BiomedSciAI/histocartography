"""This module is the main executable of the preprocessing"""

import argparse
import logging
import multiprocessing
import sys
import os
from functools import partial
from pathlib import Path
from typing import Callable, Tuple, Union

import h5py
import pandas as pd
import numpy as np
import yaml
from tqdm.auto import tqdm

from eth import ANNOTATIONS_DF, DATASET_PATH, IMAGES_DF, PREPROCESS_PATH
from feature_extraction import FeatureExtractor
from graph_builders import BaseGraphBuilder
from superpixel import SuperpixelExtractor
from stain_normalizers import StainNormalizer
from utils import (
    dynamic_import_from,
    get_next_version_number,
    read_image,
    start_logging,
    fast_histogram
)

def best_possible_assignment(annotation, superpixel):
    mask = np.empty_like(annotation)
    for pixel in pd.unique(np.ravel(superpixel)):
        assignment = np.argmax(fast_histogram(annotation[superpixel == pixel], nr_values=5))
        mask[superpixel == pixel] = assignment
    return mask


def process_best_possible_mask(
    data: Tuple[str, pd.core.series.Series],
    normalizer: Callable[[], StainNormalizer],
    superpixel_extractor: Callable[[], SuperpixelExtractor],
) -> None:
    """Process an image given the row of the metadata dataframe

    Args:
        data (Tuple[str, pd.core.series.Series]): (name, row) where row corresponds to the
            dataframe output
        normalizer (Callable[[], StainNormalizer]): Function that returns a
            StainNormalizer object
        superpixel_extractor (Callable[[], SuperpixelExtractor]): Function that returns a
            SuperpixelExtractor object
    """
    # Disable multiprocessing
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    normalizer = normalizer()
    superpixel_extractor = superpixel_extractor()

    # Read input data
    name, row = data
    image = read_image(row.path)
    annotation = read_image(row.annotation_path)

    normalized_image = normalizer.process_and_save(
        input_image=image, output_name=name
    )

    superpixels = superpixel_extractor.process_and_save(
        input_image=normalized_image, output_name=name
    )
    return name, best_possible_assignment(annotation=annotation, superpixel=superpixels)


def best_possible_masks(
    config: dict,
    cores: int = 1,
    **kwargs,
):
    """Runs the best possible masks

    Args:
        cores (int): Number of cores to use
    """
    if len(kwargs) > 0:
        logging.warning(f"Unmatched arguments: {kwargs}")
    images_metadata = pd.read_pickle(IMAGES_DF)
    annotations_metadata = pd.read_pickle(ANNOTATIONS_DF)

    annot = annotations_metadata[annotations_metadata.pathologist == 1][["path", "name"]].set_index("name")
    annot = annot.rename(columns={"path":"annotation_path"})
    images_metadata = images_metadata.join(annot)

    if not PREPROCESS_PATH.exists():
        PREPROCESS_PATH.mkdir()

    # -------------------------------------------------------------------------- STAIN NORMALIZATION
    normalizer_config = config["stain_normalizer"]
    normalizer_class = dynamic_import_from(
        "stain_normalizers", normalizer_config["class"]
    )
    normalizer = partial(
        normalizer_class,
        base_path=PREPROCESS_PATH,
        **normalizer_config.get("params", {}),
    )
    tmp_normalizer = normalizer()
    superpixel_path = tmp_normalizer.mkdir()
    if not tmp_normalizer.save_path.exists():
        target = normalizer_config["params"]["target"]
        logging.info(f"Fitting {normalizer_class.__name__} to {target}")
        target_path = images_metadata.loc[target].path
        target_image = read_image(target_path)
        tmp_normalizer.fit(target_image)
        logging.info(f"Fitting completed")

    # -------------------------------------------------------------------------- SUPERPIXEL EXTRACTION
    superpixel_config = config["superpixel_extractor"]
    superpixel_class = dynamic_import_from("superpixel", superpixel_config["class"])
    superpixel_extractor = partial(
        superpixel_class,
        base_path=superpixel_path,
        **superpixel_config.get("params", {}),
    )
    superpixels_path = superpixel_extractor().mkdir()

    worker_task = partial(
        process_best_possible_mask,
        normalizer=normalizer,
        superpixel_extractor=superpixel_extractor,
    )

    with h5py.File(superpixels_path / 'best_masks.h5', 'w') as output_file:
        if cores == 1:
            for image_metadata in tqdm(
                images_metadata.iterrows(), total=len(images_metadata)
            ):
                name, result = worker_task(image_metadata)
                output_file.create_dataset(name, data=result, compression="gzip", compression_opts=9)

        else:
            worker_pool = multiprocessing.Pool(cores)
            for name, result in tqdm(
                worker_pool.imap_unordered(
                    worker_task,
                    images_metadata.iterrows(),
                ),
                total=len(images_metadata),
                file=sys.stdout,
            ):
                output_file.create_dataset(name, data=result, compression="gzip", compression_opts=9)
            worker_pool.close()
            worker_pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default.yml")
    parser.add_argument("--level", type=str, default="WARNING")
    args = parser.parse_args()

    start_logging(args.level)
    assert Path(args.config).exists(), f"Config path does not exist: {args.config}"
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    assert (
        "upper_bound" in config
    ), f"Config does not have an entry preprocess ({config.keys()})"
    config = config["upper_bound"]

    logging.info("Start evaluating")
    assert (
        "stages" in config
    ), f"stages not defined in config {args.config}: {config.keys()}"
    assert (
        "params" in config
    ), f"params not defined in config {args.config}: {config.keys()}"
    best_possible_masks(
        config=config["stages"],
        **config["params"],
    )
