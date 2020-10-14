"""This module is the main executable of the experiment"""

import argparse
import importlib
import logging
import multiprocessing
import sys
from functools import partial
from pathlib import Path
from typing import Callable, Tuple

import pandas as pd
import yaml
from tqdm.auto import tqdm

from eth import DATASET_PATH, IMAGES_DF, PREPROCESS_PATH
from feature_extraction import FeatureExtractor
from graph_builders import BaseGraphBuilder
from superpixel import SuperpixelExtractor
from utils import (dynamic_import_from, get_next_version_number, read_image,
                   start_logging)


def process_image(
    data: Tuple[str, pd.core.series.Series],
    superpixel_extractor: Callable[[], SuperpixelExtractor],
    feature_extractor: Callable[[], FeatureExtractor],
    graph_builder: Callable[[], BaseGraphBuilder],
) -> None:
    """Process an image given the row of the metadata dataframe

    Args:
        data (Tuple[str, pd.core.series.Series]): (name, row) where row corresponds to the dataframe output
        superpixel_extractor (Callable[[], SuperpixelExtractor]): Function that returns a SuperpixelExtractor object
        feature_extractor (Callable[[], HandcraftedFeatureExtractor]): Function that returns a FeatureExtractor object
        graph_builder (Callable[[], BaseGraphBuilder]): Function that returns a BaseGraphBuilder object
        output_dir (pathlib.Path): Output directory
    """
    superpixel_extractor = superpixel_extractor()
    feature_extractor = feature_extractor()
    graph_builder = graph_builder()

    # Read input data
    name, row = data
    image = read_image(row.path)

    # TODO: stain normalization

    # Superpixels
    superpixels = superpixel_extractor.process_and_save(
        input_image=image, output_name=name
    )

    # Features
    features = feature_extractor.process_and_save(
        input_image=image, superpixels=superpixels, output_name=name
    )

    # Graph building
    graph_builder.process_and_save(
        structure=superpixels, features=features, output_name=name
    )


def preprocessing(config: dict, test: bool = False, cores: int = 1, **kwargs):
    """Runs the preprocessing with a given number of cores

    Args:
        cores (int): Number of cores to use
    """
    images_metadata = pd.read_pickle(IMAGES_DF)
    if test:
        images_metadata = images_metadata.iloc[[0, 1]]
        cores = 1
        config["superpixel_extractor"]["params"]["nr_superpixels"] = 10

    if not PREPROCESS_PATH.exists():
        PREPROCESS_PATH.mkdir()

    # -------------------------------------------------------------------------- SUPERPIXEL EXTRACTION
    superpixel_config = config["superpixel_extractor"]
    superpixel_class = dynamic_import_from("superpixel", superpixel_config["class"])
    superpixel_extractor = partial(
        superpixel_class,
        base_path=PREPROCESS_PATH,
        **superpixel_config.get("params", {}),
    )
    feature_path = superpixel_extractor().mkdir()

    # -------------------------------------------------------------------------- FEATURE EXTRACTION
    feature_config = config["feature_extractor"]
    feature_class = dynamic_import_from("feature_extraction", feature_config["class"])
    feature_extractor = partial(
        feature_class, base_path=feature_path, **feature_config.get("params", {})
    )
    graph_path = feature_extractor().mkdir()

    # -------------------------------------------------------------------------- GRAPH CONSTRUCTION
    graph_config = config["graph_builder"]
    graph_class = dynamic_import_from("graph_builders", graph_config["class"])
    graph_builder = partial(
        graph_class, base_path=graph_path, **graph_config.get("params", {})
    )
    final_path = graph_builder().mkdir()
    # --------------------------------------------------------------------------

    worker_task = partial(
        process_image,
        superpixel_extractor=superpixel_extractor,
        feature_extractor=feature_extractor,
        graph_builder=graph_builder,
    )

    if cores == 1:
        for x in tqdm(images_metadata.iterrows(), total=len(images_metadata)):
            worker_task(x)
    else:
        # Run multiprocessing
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

    # Create dataset version file
    next_version = get_next_version_number(DATASET_PATH)
    dataset_file_path = DATASET_PATH / f"v{next_version}.pickle"
    dataset_information = list()
    for file in final_path.iterdir():
        name = file.name.split(".")[0]
        assert name in images_metadata.index
        row = images_metadata.loc[name]
        dataset_information.append(
            (
                name,
                row.path,
                file,
                feature_path.name,
                graph_path.name,
                final_path.name,
                next_version,
            )
        )
    dataset_dataframe = pd.DataFrame(
        dataset_information,
        columns=[
            "name",
            "image_path",
            "graph_path",
            "superpixel_extractor",
            "feature_extractor",
            "graph_builder",
            "version",
        ],
    )
    dataset_dataframe = dataset_dataframe.set_index("name")
    dataset_dataframe.to_pickle(dataset_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preprocess", "train"])
    parser.add_argument("--config", type=str, default="default.yml")
    parser.add_argument("--test", action="store_const", const=True, default=False)
    args = parser.parse_args()

    start_logging()
    assert Path(args.config).exists(), f"Config path does not exist: {args.config}"
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    assert (
        args.command in config
    ), f"Config does not have an entry ({config.keys()}) for the desired command {args.command}"
    config = config[args.command]

    if args.command == "preprocess":
        logging.info("Start preprocessing")
        assert (
            "stages" in config
        ), f"stages not defined in config {args.config}: {config.keys()}"
        assert (
            "params" in config
        ), f"params not defined in config {args.config}: {config.keys()}"
        preprocessing(test=args.test, config=config["stages"], **config["params"])
    elif args.command == "train":
        logging.info("Training GNN")
        raise NotImplementedError
