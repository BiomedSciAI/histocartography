"""This module is the main executable of the experiment"""

import argparse
import logging
import multiprocessing
import time
from functools import partial
from math import log
import pathlib
from typing import Tuple, Callable

import dgl
from dgl.data.utils import save_graphs
import pandas as pd
from tqdm.auto import tqdm

from eth import ANNOTATIONS_DF, BASE_PATH, IMAGES_DF
from graph_builders import RAGGraphBuilder, BaseGraphBuilder
from superpixel import HandcraftedFeatureExtractor, SLICSuperpixelExtractor, SuperpixelExtractor
from utils import read_image, start_logging


def process_image(
    data: Tuple[str, pd.core.series.Series],
    superpixel_extractor: Callable[[], SuperpixelExtractor],
    feature_extractor: Callable[[], HandcraftedFeatureExtractor],
    graph_builder: Callable[[], BaseGraphBuilder],
    output_dir: pathlib.Path,
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
    superpixels = superpixel_extractor.process(image)

    # Features
    features = feature_extractor.process(image, superpixels)

    # Graph building
    graph = graph_builder(structure=superpixels, features=features)
    save_graphs(str(output_dir / f"{name}.bin"), [graph])


def preprocessing(cores: int):
    """Runs the preprocessing with a given number of cores

    Args:
        cores (int): Number of cores to use
    """
    images_metadata = pd.read_pickle(IMAGES_DF)

    graph_path = BASE_PATH / "graphs"
    if not graph_path.exists():
        graph_path.mkdir()

    # Define pipeline steps
    superpixel_extractor = partial(SLICSuperpixelExtractor, nr_superpixels=1000)
    feature_extractor = HandcraftedFeatureExtractor
    graph_builder = RAGGraphBuilder

    # Setup multiprocessing
    worker_pool = multiprocessing.Pool(cores)
    worker_task = partial(
        process_image,
        superpixel_extractor=superpixel_extractor,
        feature_extractor=feature_extractor,
        graph_builder=graph_builder,
        output_dir=graph_path,
    )

    # Run multiprocessing
    for _ in tqdm(
        worker_pool.imap_unordered(
            worker_task,
            images_metadata.iterrows(),
        ),
        total=len(images_metadata),
    ):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preprocess", "train"])
    parser.add_argument("--cores", type=int, default=1)
    args = parser.parse_args()

    start_logging()
    if args.command == "preprocess":
        logging.info("Start preprocessing")
        preprocessing(args.cores)
    elif args.command == "train":
        logging.info("Training GNN")
        raise NotImplementedError
