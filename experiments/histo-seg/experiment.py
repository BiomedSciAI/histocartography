"""This module is the main executable of the experiment"""

import argparse
import logging
import multiprocessing
import sys
from functools import partial
from pathlib import Path
from typing import Callable, Tuple, Union

import pandas as pd
import yaml
from tqdm.auto import tqdm

from eth import DATASET_PATH, IMAGES_DF, PREPROCESS_PATH
from feature_extraction import FeatureExtractor
from graph_builders import BaseGraphBuilder
from superpixel import SuperpixelExtractor
from stain_normalizers import StainNormalizer
from utils import (
    dynamic_import_from,
    get_next_version_number,
    read_image,
    start_logging,
)


def process_image(
    data: Tuple[str, pd.core.series.Series],
    normalizer: Callable[[], StainNormalizer],
    superpixel_extractor: Callable[[], SuperpixelExtractor],
    feature_extractor: Callable[[], FeatureExtractor],
    graph_builder: Callable[[], BaseGraphBuilder],
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
    normalizer = normalizer()
    superpixel_extractor = superpixel_extractor()
    feature_extractor = feature_extractor()
    graph_builder = graph_builder()

    # Read input data
    name, row = data
    image = read_image(row.path)

    if save:
        normalized_image = normalizer.process_and_save(
            input_image=image, output_name=name
        )
    else:
        normalized_image = normalizer.process(input_image=image)

    # Superpixels
    if save:
        superpixels = superpixel_extractor.process_and_save(
            input_image=normalized_image, output_name=name
        )
    else:
        superpixels = superpixel_extractor.process(input_image=normalized_image)

    # Features
    if save:
        features = feature_extractor.process_and_save(
            input_image=normalized_image, superpixels=superpixels, output_name=name
        )
    else:
        features = feature_extractor.process(
            input_image=normalized_image, superpixels=superpixels
        )

    # Graph building
    if save:
        graph_builder.process_and_save(
            structure=superpixels, features=features, output_name=name
        )
    else:
        graph_builder.process(structure=superpixels, features=features)


def preprocessing(
    config: dict,
    test: bool = False,
    save: bool = True,
    cores: int = 1,
    subsample: Union[None, int] = None,
    **kwargs,
):
    """Runs the preprocessing with a given number of cores

    Args:
        cores (int): Number of cores to use
    """
    if len(kwargs) > 0:
        logging.warning(f"Unmatched arguments: {kwargs}")
    images_metadata = pd.read_pickle(IMAGES_DF)

    if save and not PREPROCESS_PATH.exists():
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
    if save:
        tmp_normalizer = normalizer()
        superpixel_path = tmp_normalizer.mkdir()
        if not tmp_normalizer.save_path.exists():
            target = normalizer_config["params"]["target"]
            logging.info("Fitting {normalizer_class.__name__} to {target}")
            target_path = images_metadata.loc[target].path
            target_image = read_image(target_path)
            tmp_normalizer.fit(target_image)
    else:
        superpixel_path = None

    # -------------------------------------------------------------------------- SUPERPIXEL EXTRACTION
    superpixel_config = config["superpixel_extractor"]
    superpixel_class = dynamic_import_from("superpixel", superpixel_config["class"])
    superpixel_extractor = partial(
        superpixel_class,
        base_path=superpixel_path,
        **superpixel_config.get("params", {}),
    )
    feature_path = superpixel_extractor().mkdir() if save else None

    # -------------------------------------------------------------------------- FEATURE EXTRACTION
    feature_config = config["feature_extractor"]
    feature_class = dynamic_import_from("feature_extraction", feature_config["class"])
    feature_extractor = partial(
        feature_class, base_path=feature_path, **feature_config.get("params", {})
    )
    graph_path = feature_extractor().mkdir() if save else None

    # -------------------------------------------------------------------------- GRAPH CONSTRUCTION
    graph_config = config["graph_builder"]
    graph_class = dynamic_import_from("graph_builders", graph_config["class"])
    graph_builder = partial(
        graph_class, base_path=graph_path, **graph_config.get("params", {})
    )
    final_path = graph_builder().mkdir() if save else None
    # --------------------------------------------------------------------------

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
            images_metadata.iterrows(), total=len(images_metadata)
        ):
            worker_task(image_metadata)
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
    if save:
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
    parser.add_argument("--subsample", type=int, default=-1)
    parser.add_argument("--test", action="store_const", const=True, default=False)
    parser.add_argument("--nosave", action="store_const", const=True, default=False)
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
        preprocessing(
            test=args.test,
            config=config["stages"],
            save=(not args.nosave),
            subsample=None if args.subsample < 0 else args.subsample,
            **config["params"],
        )
    elif args.command == "train":
        logging.info("Training GNN")
        raise NotImplementedError
