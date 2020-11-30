"""Preprocessing utilities"""
import importlib
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import h5py
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


class PipelineRunner:
    def __init__(
        self,
        output_path: str,
        inputs: Iterable[str] = [],
        outputs: Iterable[str] = [],
        stages: Iterable[dict] = [],
        save: bool = True,
    ):
        """Create a pipeline runner for a given configuration

        Args:
            output_path (str): Path to output the intermediate files
            inputs (Iterable[str], optional): Inputs to the pipeline. Defaults to [].
            outputs (Iterable[str], optional): Outputs of the pipeline. Defaults to [].
            stages (Iterable[dict], optional): Stages to complete. Defaults to [].
            save (bool, optional): Whether to save the results. Defaults to True.
        """
        self.inputs = inputs
        self.outputs = outputs
        self.save = save
        self.stages = list()
        self.stage_configs = list()
        path = output_path if save else None
        for stage in stages:
            name, config = list(stage.items())[0]
            print(name, config)
            stage_class = dynamic_import_from(
                f"histocartography.preprocessing.{name}", config.pop("class")
            )
            pipeline_stage = partial(
                stage_class,
                base_path=path,
                **config.pop("params", {}),
            )
            self.stages.append(pipeline_stage())
            self.stage_configs.append(config)
            path = pipeline_stage().mkdir() if save else None

    def run(self, name: Optional[str], **inputs):
        # Validate inputs
        assert (
            not self.save or name is not None
        ), "Either specify save=False or provide a name"
        for input_name in self.inputs:
            assert input_name in inputs, f"{input_name} not found in keyword arguments"

        # Compute pipelines steps
        variables = deepcopy(inputs)
        for stage, config in zip(self.stages, self.stage_configs):
            step_input = [variables[k] for k in config["inputs"]]
            step_output = stage.process_and_save(name, *step_input)
            if not isinstance(step_output, tuple):
                step_output = tuple([step_output])
            assert len(step_output) == len(config["outputs"]), (
                f"Number of outputs in config mismatches actual number of outputs in {stage.__class__.__name__}"
                f"Got {len(step_output)} outputs of type {list(map(type, step_output))},"
                f"but expected {len(config['outputs'])} outputs"
            )
            for key, value in zip(config["outputs"], step_output):
                variables[key] = value

        # Handle output
        for output_name in self.outputs:
            assert (
                output_name in variables
            ), f"{output_name} should be returned, but was never computed"
        return {k: variables[k] for k in self.outputs}


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
