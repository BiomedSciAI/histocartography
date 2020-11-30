"""Preprocessing utilities"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union
import h5py
import importlib
import numpy as np


class PipelineStep(ABC):
    """Base pipelines step"""

    def __init__(self, base_path: Union[None, str, Path] = None) -> None:
        """Abstract class that helps with saving and loading precomputed results

        Args:
            base_path (Union[None, str, Path], optional): Base path to save results.
                Defaults to None.
        """
        name = self.__repr__()
        self.base_path = base_path
        if self.base_path is not None:
            self.output_dir = Path(self.base_path) / name
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
        ), "Can only create directory if base_path was not None when constructing the object"
        if not self.output_dir.exists():
            self.output_dir.mkdir()
        return self.output_dir

    @abstractmethod
    def process(self, **kwargs: Any) -> Any:
        """Process an input"""

    def process_and_save(self, output_name: str, *args, **kwargs: Any) -> Any:
        """Process and save in the provided path as as .h5 file

        Args:
            output_name (str): Name of output file
        """
        assert (
            self.base_path is not None
        ), "Can only save intermediate output if base_path was not None when constructing the object"
        output_path = self.output_dir / f"{output_name}.h5"
        if output_path.exists():
            logging.info(
                f"{self.__class__.__name__}: Output of {output_name} already exists, using it instead of recomputing"
            )
            with h5py.File(output_path, "r") as input_file:
                output = input_file[self.output_key][()]
        else:
            output = self.process(*args, **kwargs)
            with h5py.File(output_path, "w") as output_file:
                output_file.create_dataset(
                    self.output_key, data=output, compression="gzip", compression_opts=9
                )
        return output


def fast_histogram(input_array: np.ndarray, nr_values: int) -> np.ndarray:
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
