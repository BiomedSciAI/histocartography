"""This module is the main executable of the preprocessing"""

import argparse
import logging
import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable, Tuple, Optional

import pandas as pd
import yaml
from tqdm.auto import tqdm

from eth import ANNOTATIONS_DF, IMAGES_DF, PREPROCESS_PATH

from histocartography.preprocessing.feature_extraction import FeatureExtractor
from histocartography.preprocessing.graph_builders import BaseGraphBuilder
from histocartography.preprocessing.stain_normalizers import StainNormalizer
from histocartography.preprocessing.superpixel import SuperpixelExtractor
from utils import (
    dynamic_import_from,
    merge_metadata,
    read_image,
    start_logging,
)


def process_image(
    data: Tuple[str, pd.core.series.Series],
    normalizer: Callable[[], StainNormalizer],
    superpixel_extractor: Optional[Callable[[], SuperpixelExtractor]],
    feature_extractor: Optional[Callable[[], FeatureExtractor]],
    graph_builder: Optional[Callable[[], BaseGraphBuilder]],
    save: bool,
) -> None:
    """Process an image given the row of the metadata dataframe

    Args:
        data (Tuple[str, pd.core.series.Series]): (name, row) where row corresponds to the
            dataframe output
        normalizer (Callable[[], StainNormalizer]): Function that returns a
            StainNormalizer object
        superpixel_extractor (Callable[[], SuperpixelExtractor]): Function that returns a
            SuperpixelExtractor object
        feature_extractor (Callable[[], HandcraftedFeatureExtractor]): Function that returns a
            FeatureExtractor object
        graph_builder (Callable[[], BaseGraphBuilder]): Function that returns a BaseGraphBuilder
            object
        output_dir (pathlib.Path): Output directory
    """
    # Disable multiprocessing
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # Read input data
    name, row = data
    image = read_image(row.path)

    normalizer = normalizer()
    if save:
        normalized_image = normalizer.process_and_save(
            input_image=image, output_name=name
        )
    else:
        normalized_image = normalizer.process(input_image=image)

    # Superpixels
    if superpixel_extractor is not None:
        superpixel_extractor = superpixel_extractor()
        if save:
            superpixels = superpixel_extractor.process_and_save(
                input_image=normalized_image, output_name=name
            )
        else:
            superpixels = superpixel_extractor.process(input_image=normalized_image)

        # Features
        if feature_extractor is not None:
            feature_extractor = feature_extractor()
            if save:
                features = feature_extractor.process_and_save(
                    input_image=normalized_image, instance_map=superpixels, output_name=name
                )
            else:
                features = feature_extractor.process(
                    input_image=normalized_image, instance_map=superpixels
                )

            if graph_builder is not None:
                graph_builder = graph_builder()

                # Optional annotation loading
                if "annotation_path" in row:
                    annotation = read_image(row.annotation_path)
                else:
                    annotation = None

                if save:
                    graph_builder.process_and_save(
                        structure=superpixels,
                        features=features,
                        annotation=annotation,
                        output_name=name,
                    )
                else:
                    graph_builder.process(
                        structure=superpixels, features=features, annotation=annotation
                    )


def get_pipeline_step(stage, config, path):
    stage_class = dynamic_import_from(
        f"histocartography.preprocessing.{stage}", config["class"]
    )
    pipeline_stage = partial(
        stage_class,
        base_path=path,
        **config.get("params", {}),
    )
    return pipeline_stage

def preprocessing(
    config: dict,
    test: bool = False,
    save: bool = True,
    cores: int = 1,
    labels: bool = False,
    subsample: Optional[int] = None,
    **kwargs,
):
    """Runs the preprocessing with a given number of cores

    Args:
        cores (int): Number of cores to use
    """
    if len(kwargs) > 0:
        logging.warning(f"Unmatched arguments: {kwargs}")
    images_metadata = pd.read_pickle(IMAGES_DF)
    if labels:
        annotation_metadata = pd.read_pickle(ANNOTATIONS_DF)
        images_metadata = merge_metadata(images_metadata, annotation_metadata)

    if save and not PREPROCESS_PATH.exists():
        PREPROCESS_PATH.mkdir()

    normalizer = get_pipeline_step("stain_normalizers", config["stain_normalizer"], PREPROCESS_PATH)
    tmp_normalizer = normalizer()
    superpixel_path = tmp_normalizer.mkdir() if save else None
    if not tmp_normalizer.save_path.exists():
        target = config["stain_normalizer"]["params"]["target"]
        logging.info(f"Fitting normalizer to {target}")
        target_path = images_metadata.loc[target].path
        target_image = read_image(target_path)
        tmp_normalizer.fit(target_image)
        logging.info(f"Fitting completed")

    graph_builder = None
    feature_extractor = None
    superpixel_extractor = None

    superpixel_config = config["superpixel_extractor"]
    if superpixel_config is not None:
        superpixel_extractor = get_pipeline_step("superpixel", superpixel_config, superpixel_path)
        feature_path = superpixel_extractor().mkdir() if save else None
        feature_config = config["feature_extractor"]
        if feature_config is not None:
            feature_extractor = get_pipeline_step("feature_extraction", feature_config, feature_path)
            graph_path = feature_extractor().mkdir() if save else None
            graph_config = config["graph_builder"]
            if graph_config is not None:
                graph_builder = get_pipeline_step("graph_builders", graph_config, graph_path)
                graph_builder().mkdir() if save else None

    worker_task = partial(
        process_image,
        normalizer=normalizer,
        superpixel_extractor=superpixel_extractor,
        feature_extractor=feature_extractor,
        graph_builder=graph_builder,
        save=save,
    )

    # Handle test mode
    if test:
        images_metadata = images_metadata.iloc[[0]]
        cores = 1
        config["superpixel_extractor"]["params"]["nr_superpixels"] = 10

    # Handle subsample mode
    if subsample is not None:
        assert (
            1 <= subsample <= len(images_metadata)
        ), f"Subsample needs to be larger than 1 and smaller than the number of rows but is {subsample}"
        images_metadata = images_metadata.sample(subsample)

    if cores == 1:
        for image_metadata in tqdm(
            images_metadata.iterrows(),
            total=len(images_metadata),
            file=sys.stdout,
        ):
            worker_task(image_metadata)
    else:
        worker_pool = multiprocessing.Pool(cores)
        for _ in tqdm(
            worker_pool.imap_unordered(
                worker_task,
                images_metadata.iterrows(),
            ),
            total=len(images_metadata),
            file=sys.stdout,
        ):
            pass
        worker_pool.close()
        worker_pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default.yml")
    parser.add_argument("--level", type=str, default="WARNING")
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--test", action="store_const", const=True, default=False)
    parser.add_argument("--nosave", action="store_const", const=True, default=False)
    args = parser.parse_args()

    start_logging(args.level)
    assert Path(args.config).exists(), f"Config path does not exist: {args.config}"
    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    assert (
        "preprocess" in config
    ), f"Config does not have an entry preprocess ({config.keys()})"
    config = config["preprocess"]

    logging.info("Start preprocessing")
    assert (
        "stages" in config
    ), f"stages not defined in config {args.config}: {config.keys()}"
    assert (
        "params" in config
    ), f"params not defined in config {args.config}: {config.keys()}"
    preprocessing(
        test=args.test,
        config=config["stages"],
        save=(not args.nosave),
        subsample=None if args.subsample < 0 else args.subsample,
        **config["params"],
    )
