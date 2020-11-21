import argparse
import logging
import multiprocessing
import os
import sys
from functools import partial
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from dgl.data.utils import load_graphs, save_graphs
from torch import mean, normal
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from tqdm.auto import tqdm

from constants import CENTROID, FEATURES, GNN_NODE_FEAT_IN, LABEL
from eth import NR_CLASSES
from feature_extraction import HandcraftedFeatureExtractor
from graph_builders import RAGGraphBuilder
from superpixel import SLICSuperpixelExtractor
from utils import start_logging

with os.popen("hostname") as subprocess:
    hostname = subprocess.read()
if hostname.startswith("zhcc"):
    BASE_PATH = Path("/dataT/anv/Data/MNIST")
elif hostname.startswith("Gerbil"):
    BASE_PATH = Path("/Users/anv/Documents/MNIST")
else:
    assert False, f"Hostname {hostname} does not have the data path specified"
if not BASE_PATH.exists():
    BASE_PATH.mkdir()
BACKGROUND_CLASS = 10
NR_CLASSES = 10


def process_element(
    data, superpixel_extractor, feature_extractor, graph_builder, output_dir
):
    # Disable multiprocessing
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    superpixel_extractor = superpixel_extractor()
    feature_extractor = feature_extractor()
    graph_builder = graph_builder()

    i, (image, label) = data
    image = np.array(image)
    label_mask = np.ones_like(image) * BACKGROUND_CLASS
    label_mask[image > 0] = label
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    superpixels = superpixel_extractor.process(image)
    features = feature_extractor.process(image, superpixels)
    graph = graph_builder.process(superpixels, features, label_mask)
    save_graphs(output_dir + f"_{i}.bin", graph)


def cleanup(output_path, output_dir, name):
    all_graphs = list()
    for file in tqdm(glob(output_path + "*"), desc=f"Aggregating {name}"):
        all_graphs.append(load_graphs(file)[0][0])
    save_graphs(str(output_dir / f"all_{name}.bin"), all_graphs)
    for file in tqdm(glob(output_path + "*"), desc=f"Cleaning up {name}"):
        os.remove(file)


def process_parallel(nr_superpixels, cores, train, output_dir):
    superpixel_extractor = partial(
        SLICSuperpixelExtractor, nr_superpixels=nr_superpixels
    )
    feature_extractor = HandcraftedFeatureExtractor
    graph_builder = partial(RAGGraphBuilder, kernel_size=3, nr_classes=NR_CLASSES + 1)
    output_path = str(
        output_dir / f"mnist_{'train' if train else 'valid'}_{nr_superpixels}"
    )
    worker_task = partial(
        process_element,
        superpixel_extractor=superpixel_extractor,
        feature_extractor=feature_extractor,
        graph_builder=graph_builder,
        output_dir=output_path,
    )
    dataset = MNIST(output_dir, train=train, download=True)
    if cores == 1:
        for el in tqdm(dataset):
            worker_task(el)
    else:
        with multiprocessing.Pool(cores) as worker_pool:
            for _ in tqdm(
                worker_pool.imap_unordered(
                    worker_task, enumerate(dataset), chunksize=10
                ),
                desc=f"Compute {'train' if train else 'valid'} {nr_superpixels}",
                total=len(dataset),
                file=sys.stdout,
            ):
                pass
    cleanup(
        output_path=output_path,
        output_dir=output_dir,
        name=f"{'train' if train else 'valid'}_{nr_superpixels}",
    )


def compute_normalizer(name, nr_superpixels, **kwargs):
    dataset = MNISTDataset(
        str(BASE_PATH / f"all_{name}_{nr_superpixels}.bin"),
        centroid_features="no",
    )
    all_features = list()
    for i in range(len(dataset)):
        all_features.append(dataset[i][0].ndata[GNN_NODE_FEAT_IN].mean(axis=0))
    all_features = torch.cat(all_features)
    torch.save(all_features.mean(axis=0), BASE_PATH / f"{name}_{nr_superpixels}_normalizer_mean.pth")
    torch.save(all_features.std(axis=0), BASE_PATH / f"{name}_{nr_superpixels}_normalizer_std.pth")


class MNISTDataset(Dataset):
    def __init__(self, bin_file, centroid_features, mean=None, std=None) -> None:
        self.mean = mean
        self.std = std
        self.graphs = load_graphs(bin_file)[0]
        self.graph_labels = list()
        for graph in self.graphs:
            candidates = pd.unique(np.ravel(graph.ndata[LABEL]))
            label = max(candidates[candidates != BACKGROUND_CLASS])
            self.graph_labels.append(
                torch.nn.functional.one_hot(
                    torch.Tensor([label]).to(torch.int64), NR_CLASSES
                ).to(torch.int8)[0]
            )
        self.image_sizes = [(28, 28)] * len(self.graph_labels)
        self._select_graph_features(centroid_features)

    def _select_graph_features(self, centroid_features, normalizer=None):
        for graph, image_size in zip(self.graphs, self.image_sizes):
            if centroid_features == "only":
                features = (graph.ndata[CENTROID] / torch.Tensor(image_size)).to(
                    torch.float32
                )
                graph.ndata.pop(FEATURES)
            else:
                features = graph.ndata.pop(FEATURES).to(torch.float32)
                if self.mean is not None and self.std is not None:
                    features = (features - self.mean) / self.std

                if centroid_features == "cat":
                    features = torch.cat(
                        [
                            features,
                            (graph.ndata[CENTROID] / torch.Tensor(image_size)).to(
                                torch.float32
                            ),
                        ],
                        dim=1,
                    )
            graph.ndata[GNN_NODE_FEAT_IN] = features

    def __getitem__(self, index: int):
        graph = self.graphs[index]
        node_labels = graph.ndata[LABEL]
        graph_label = self.graph_labels[index]
        return graph, graph_label, node_labels

    def __len__(self) -> int:
        return len(self.graph_labels)


def prepare_graph_datasets(centroid_features, nr_superpixels, normalize_features=False):
    if normalize_features:
        mean = torch.load(BASE_PATH / f"train_{nr_superpixels}_normalizer_mean.pth")
        std = torch.load(BASE_PATH / f"train_{nr_superpixels}_normalizer_std.pth")

    else:
        mean = None
        std = None
    train_dataset = MNISTDataset(
        str(BASE_PATH / f"all_train_{nr_superpixels}.bin"),
        centroid_features=centroid_features,
        mean=mean,
        std=std,
    )
    valid_dataset = MNISTDataset(
        str(BASE_PATH / f"all_valid_{nr_superpixels}.bin"),
        centroid_features=centroid_features,
        mean=mean,
        std=std,
    )
    return train_dataset, valid_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="mnist.yml")
    parser.add_argument("--level", type=str, default="WARNING")
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
    process_parallel(train=True, output_dir=BASE_PATH, **config["params"])
    process_parallel(train=False, output_dir=BASE_PATH, **config["params"])
    compute_normalizer(name="train", **config["params"])
