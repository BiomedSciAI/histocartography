"""This module handles helper general functions"""
import importlib
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import dgl
import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from PIL import Image


class PipelineStep(ABC):
    """Base pipelines step"""

    def __init__(self, base_path: Union[None, str, Path] = None) -> None:
        """Abstract class that helps with saving and loading precomputed results

        Args:
            base_path (Union[None, str, Path], optional): Base path to save results. Defaults to None.
        """
        name = self.__repr__()
        self.base_path = base_path
        if self.base_path is not None:
            self.output_dir = Path(base_path) / name
            self.output_key = "default_key"

    def __repr__(self) -> str:
        """Representation of a graph builder

        Returns:
            str: Representation of a graph builder
        """
        variables = ",".join([f"{k}={v}" for k, v in sorted(self.__dict__.items())])
        return f"{self.__class__.__name__}({variables})"

    def mkdir(self) -> Path:
        """Create path to output files"""
        assert (
            self.base_path is not None
        ), f"Can only create directory if base_path was not None when constructing the object"
        if not self.output_dir.exists():
            self.output_dir.mkdir()
        return self.output_dir

    @abstractmethod
    def process(self, **kwargs) -> Any:
        """Process an input"""

    def process_and_save(self, output_name: str, **kwargs) -> Any:
        """Process and save in the provided path as as .h5 file

        Args:
            output_name (str): Name of output file
        """
        assert (
            self.base_path is not None
        ), f"Can only save intermediate output if base_path was not None when constructing the object"
        output_path = self.output_dir / f"{output_name}.h5"
        if output_path.exists():
            logging.info(
                f"Output of {output_name} already exists, using it instead of recomputing"
            )
            input_file = h5py.File(output_path, "r")
            output = input_file[self.output_key][()]
            input_file.close()
        else:
            output = self.process(**kwargs)
            output_file = h5py.File(output_path, "w")
            output_file.create_dataset(self.output_key, data=output)
            output_file.close()
        return output


def fast_mode(input_array: np.array, nr_values, axis: int = 0) -> np.array:
    """Calculates the mode of an tensor over an axis where only values from 0 up to (excluding) nr_values occur.

    Args:
        x (np.array): Input Tensor
        nr_valuesint (int): Possible values. From 0 up to (exclusing) nr_values.
        axis (int, optional): Axis to do the mode over. Defaults to 0.

    Returns:
        np.array: Output Tensor
    """
    output_array = np.empty((nr_values, input_array.shape[1], input_array.shape[2]))
    for i in range(nr_values):
        output_array[i, ...] = (input_array == i).sum(axis=axis)
    return np.argmax(output_array, axis=0)


def fast_histogram(input_array: np.array, nr_values: int) -> np.array:
    """Calculates a histogram of a matrix of the values from 0 up to (excluding) nr_values

    Args:
        x (np.array): Input tensor
        nr_values (int): Possible values. From 0 up to (exclusing) nr_values.

    Returns:
        np.array: Output tensor
    """
    output_array = np.empty(nr_values, dtype=int)
    for i in range(nr_values):
        output_array[i] = (input_array == i).sum()
    return output_array


def read_image(image_path: str) -> np.array:
    """Reads an image from a path and converts it into a numpy array

    Args:
        image_path (str): Path to the image

    Returns:
        np.array: A numpy array representation of the image
    """
    assert image_path.exists()
    img = Image.open(image_path)
    image = np.array(img)
    img.close()
    return image


def start_logging(level="INFO") -> None:
    """Start logging with the standard format

    Args:
        level (str, optional): Logging level. Defaults to "INFO".
    """
    logging.basicConfig(
        level=level,
        format="%(levelname)s - %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Start logging")


def show_superpixel_heatmap(superpixels: np.array) -> None:
    """Show a heatmap of the provided superpixels

    Args:
        superpixels (np.array): Superpixels
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    sns.heatmap(superpixels - 1, annot=True, fmt="d", ax=ax, square=True, cbar=False)
    ax.set_axis_off()
    fig.show()


def show_graph(graph: dgl.DGLGraph) -> None:
    """Show DGL graph with Kamanda Kawai Layout

    Args:
        graph (dgl.DGLGraph): Graph to show
    """
    nx_G = graph.to_networkx().to_undirected()
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, node_color=[[0.7, 0.7, 0.7]])


def get_next_version_number(path: Path) -> int:
    """Get next ascending version number

    Args:
        path (Path): Path to check for prior versions

    Returns:
        int: Next version number
    """
    if not path.exists():
        path.mkdir()
    existing = list(
        map(lambda x: int(re.findall(r"[0-9]+", x.name)[0]), path.iterdir())
    )
    if len(existing) == 0:
        return 0
    else:
        return max(existing) + 1


def dynamic_import_from(source_file: str, class_name: str) -> Any:
    """Do a from source_file import class_name dynamically

    Args:
        source_file (str): Where to import from
        class_name (str): What to import

    Returns:
        Any: The class to be imported
    """
    module = importlib.import_module(source_file)
    return getattr(module, class_name)
