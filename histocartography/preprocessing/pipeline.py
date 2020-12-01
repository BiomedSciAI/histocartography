"""Pipeline utilities"""
import logging
import multiprocessing
import os
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import h5py
import pandas as pd
from histocartography.utils import dynamic_import_from
from tqdm.auto import tqdm


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

    def precompute(self) -> None:
        """Precompute all necessary information for this step"""
        pass

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

    def precompute(self) -> None:
        """Precompute all necessary information for all stages"""
        for stage in self.stages:
            stage.precompute()

    def run(self, name: Optional[str], **inputs: Any) -> Dict[str, Any]:
        """Run the preprocessing pipeline for a given name and input parameters and return the specified outputs

        Args:
            name (Optional[str]): Optional name to use for saving

        Returns:
            Dict[str, Any]: Specified outputs
        """

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
            if self.save:
                step_output = stage.process_and_save(name, *step_input)
            else:
                step_output = stage.process(*step_input)
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


class BatchPipelineRunner:
    def __init__(
        self, pipeline_config: Dict[str, Any], output_path: str, save: bool = True
    ) -> None:
        """Run Helper that runs the pipeline for multiple inputs in a multiprocessed fashion.

        Args:
            pipeline_config (Dict[str, Any]): Configuration of the pipeline
            output_path (str): Path to save the outputs to
            save (bool, optional): Whether to save the outputs. Defaults to True.
        """
        self.pipeline_config = pipeline_config
        self.output_path = output_path
        self.save = save

    def _build_pipeline_runner(self) -> PipelineRunner:
        """Builds and returns a PipelineRunner with the correct configuration

        Returns:
            PipelineRunner: Runner object
        """
        config = deepcopy(self.pipeline_config)
        return PipelineRunner(output_path=self.output_path, save=self.save, **config)

    def _worker_task(self, data: Tuple[Any, pd.core.series.Series]) -> None:
        """Runs the task of a single worker

        Args:
            data (Tuple[Any, pd.core.series.Series]): The index and row of the dataframe,
                                                      as returned from df.iterrows()
        """
        # Disable multiprocessing
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

        name, row = data
        pipeline = self._build_pipeline_runner()
        pipeline.run(name=name, **row)

    def _precompute(self):
        """Precompute all necessary information for all stages"""
        self._build_pipeline_runner().precompute()

    def run(self, metadata: pd.DataFrame, cores: int = 1) -> None:
        """Runs the pipeline for the provided metadata dataframe and a specified
           number of cores for multiprocessing.

        Args:
            metadata (pd.DataFrame): Dataframe with the columns as defined in the config inputs
            cores (int, optional): Number of cores to use for multiprocessing. Defaults to 1.
        """
        self._precompute()
        if cores == 1:
            pipeline = self._build_pipeline_runner()
            for name, row in tqdm(
                metadata.iterrows(), total=len(metadata), file=sys.stdout
            ):
                pipeline.run(name=name, **row)
        else:
            worker_pool = multiprocessing.Pool(cores)
            for _ in tqdm(
                worker_pool.imap_unordered(
                    self._worker_task,
                    metadata.iterrows(),
                ),
                total=len(metadata),
                file=sys.stdout,
            ):
                pass
            worker_pool.close()
            worker_pool.join()
