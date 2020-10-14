"""This module is the main executable of the experiment"""

import argparse
import logging
import multiprocessing
import sys
from functools import partial
from typing import Callable, Tuple

import pandas as pd
from tqdm.auto import tqdm

from eth import DATASET_PATH, IMAGES_DF, PREPROCESS_PATH
from feature_extraction import HandcraftedFeatureExtractor
from graph_builders import BaseGraphBuilder, RAGGraphBuilder
from superpixel import SLICSuperpixelExtractor, SuperpixelExtractor
from utils import get_next_version_number, read_image, start_logging


def process_image(
    data: Tuple[str, pd.core.series.Series],
    superpixel_extractor: Callable[[], SuperpixelExtractor],
    feature_extractor: Callable[[], HandcraftedFeatureExtractor],
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


def preprocessing(cores: int, nr_superpixels: int, test: bool = False):
    """Runs the preprocessing with a given number of cores

    Args:
        cores (int): Number of cores to use
    """
    images_metadata = pd.read_pickle(IMAGES_DF)
    if test:
        images_metadata = images_metadata.iloc[[0, 1]]
        nr_superpixels = 10

    if not PREPROCESS_PATH.exists():
        PREPROCESS_PATH.mkdir()

    # Define pipeline steps
    superpixel_extractor = partial(
        SLICSuperpixelExtractor,
        nr_superpixels=nr_superpixels,
        base_path=PREPROCESS_PATH,
    )
    feature_path = superpixel_extractor().mkdir()

    feature_extractor = partial(HandcraftedFeatureExtractor, base_path=feature_path)
    graph_path = feature_extractor().mkdir()

    graph_builder = partial(RAGGraphBuilder, base_path=graph_path)
    final_path = graph_builder().mkdir()

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
        worker_pool.join()
        worker_pool.close()

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
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--nr_superpixels", type=int, default=1000)
    parser.add_argument("--test", action="store_const", const=True, default=False)
    args = parser.parse_args()

    start_logging()
    if args.command == "preprocess":
        logging.info("Start preprocessing")
        preprocessing(
            cores=args.cores, nr_superpixels=args.nr_superpixels, test=args.test
        )
    elif args.command == "train":
        logging.info("Training GNN")
        raise NotImplementedError
