"""Pipeline utilities"""
import logging
import multiprocessing
import os
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import h5py
import pandas as pd
from tqdm.auto import tqdm

from histocartography.utils import dynamic_import_from, signal_last


class PipelineStep(ABC):
    """Base pipelines step"""

    def __init__(
        self,
        save_path: Union[None, str, Path] = None,
        precompute: bool = True,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Abstract class that helps with saving and loading precomputed results

        Args:
            save_path (Union[None, str, Path], optional): Base path to save results to.
                When set to None, the results are not saved to disk. Defaults to None.
            precompute (bool, optional): Whether to perform the precomputation necessary
                for the step. Defaults to True.
            link_path (Union[None, str, Path], optional): Path to link the output directory
                to. When None, no link is created. Only supported when save_path is not None.
                Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save the output of
                the precomputation to. If not specified it defaults to the output directory
                of the step when save_path is not None. Defaults to None.
        """
        assert (
            save_path is not None or link_path is None
        ), "link_path only supported when save_path is not None"

        name = self.__repr__()
        self.save_path = save_path
        if self.save_path is not None:
            self.output_dir = Path(self.save_path) / name
            self.output_key = "default_key"
            self._mkdir()
            if precompute_path is None:
                precompute_path = save_path

        if precompute:
            self.precompute(
                link_path=link_path,
                precompute_path=precompute_path)

    def __repr__(self) -> str:
        """Representation of a pipeline step.

        Returns:
            str: Representation of a pipeline step.
        """
        variables = ",".join(
            [f"{k}={v}" for k, v in sorted(self.__dict__.items())])
        return (
            f"{self.__class__.__name__}({variables})".replace(" ", "")
            .replace('"', "")
            .replace("'", "")
            .replace("..", "")
            .replace("/", "_")
        )

    def _mkdir(self) -> None:
        """Create path to output files"""
        assert (
            self.save_path is not None
        ), "Can only create directory if base_path was not None when constructing the object"
        if not self.output_dir.exists():
            self.output_dir.mkdir()

    def _link_to_path(self, link_directory: Union[None, str, Path]) -> None:
        """Links the output directory to the given directory.

        Args:
            link_directory (Union[None, str, Path]): Directory to link to
        """
        if link_directory is None or Path(
                link_directory).parent.resolve() == Path(self.output_dir):
            logging.info("Link to self skipped")
            return
        assert (
            self.save_path is not None
        ), f"Linking only supported when saving is enabled, i.e. when save_path is passed in the constructor."
        if os.path.islink(link_directory):
            if os.path.exists(link_directory):
                logging.info("Link already exists: overwriting...")
                os.remove(link_directory)
            else:
                logging.critical(
                    "Link exists, but points nowhere. Ignoring...")
                return
        elif os.path.exists(link_directory):
            os.remove(link_directory)
        os.symlink(self.output_dir, link_directory, target_is_directory=True)

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Precompute all necessary information for this step

        Args:
            link_path (Union[None, str, Path], optional): Path to link the output to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to load/save the precomputation outputs. Defaults to None.
        """
        pass

    def process(
        self, *args: Any, output_name: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Main process function of the step and outputs the result. Try to saves the output when output_name is passed.

        Args:
            output_name (Optional[str], optional): Unique identifier of the passed datapoint. Defaults to None.

        Returns:
            Any: Result of the pipeline step
        """
        if output_name is not None and self.save_path is not None:
            return self._process_and_save(
                *args, output_name=output_name, **kwargs)
        else:
            return self._process(*args, **kwargs)

    @abstractmethod
    def _process(self, *args: Any, **kwargs: Any) -> Any:
        """Abstract method that performs the computation of the pipeline step

        Returns:
            Any: Result of the pipeline step
        """

    def _get_outputs(self, input_file: h5py.File) -> Union[Any, Tuple]:
        """Extracts the step output from a given h5 file

        Args:
            input_file (h5py.File): File to load from

        Returns:
            Union[Any, Tuple]: Previously computed output of the step
        """
        outputs = list()
        nr_outputs = len(input_file.keys())

        # Legacy, remove at some point
        if nr_outputs == 1 and self.output_key in input_file.keys():
            return tuple([input_file[self.output_key][()]])

        for i in range(nr_outputs):
            outputs.append(input_file[f"{self.output_key}_{i}"][()])
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

    def _set_outputs(self, output_file: h5py.File,
                     outputs: Union[Tuple, Any]) -> None:
        """Save the step output to a given h5 file

        Args:
            output_file (h5py.File): File to write to
            outputs (Union[Tuple, Any]): Computed step output
        """
        if not isinstance(outputs, tuple):
            outputs = tuple([outputs])
        for i, output in enumerate(outputs):
            output_file.create_dataset(
                f"{self.output_key}_{i}",
                data=output,
                compression="gzip",
                compression_opts=9,
            )

    def _process_and_save(
        self, *args: Any, output_name: str, **kwargs: Any
    ) -> Any:
        """Process and save in the provided path as as .h5 file

        Args:
            output_name (str): Unique identifier of the the passed datapoint

        Raises:
            read_error (OSError): When the unable to read to self.output_dir/output_name.h5
            write_error (OSError): When the unable to write to self.output_dir/output_name.h5
        Returns:
            Any: Result of the pipeline step
        """
        assert (
            self.save_path is not None
        ), "Can only save intermediate output if base_path was not None when constructing the object"
        output_path = self.output_dir / f"{output_name}.h5"
        if output_path.exists():
            logging.info(
                f"{self.__class__.__name__}: Output of {output_name} already exists, using it instead of recomputing"
            )
            try:
                with h5py.File(output_path, "r") as input_file:
                    output = self._get_outputs(input_file=input_file)
            except OSError as read_error:
                print(f"\n\nCould not read from {output_path}!\n\n")
                raise read_error
        else:
            output = self._process(*args, **kwargs)
            try:
                with h5py.File(output_path, "w") as output_file:
                    self._set_outputs(output_file=output_file, outputs=output)
            except OSError as write_error:
                print(f"\n\nCould not write to {output_path}!\n\n")
                raise write_error
        return output


class PipelineRunner:
    def __init__(
        self,
        output_path: Optional[str] = None,
        inputs: Optional[Iterable[str]] = None,
        outputs: Optional[Iterable[str]] = None,
        stages: Iterable[dict] = [],
        save_intermediate: bool = False,
        precompute: bool = True,
    ) -> None:
        """Create a pipeline runner for a given configuration

        Args:
            output_path (Optional[str], optional): Path to the output and intermediate files.
                When set to None the runner does not save the outputs. Defaults to None.
            inputs (Optional[Iterable[str]], optional): Inputs to the pipeline. Defaults to None.
            outputs (Optional[Iterable[str]], optional): Outputs of the pipeline. Defaults to None.
            stages (Iterable[dict], optional): Stages to complete. Defaults to [].
            save_intermediate (bool, optional): Whether to save the intermediate steps. Defaults to False.
            precompute (bool, optional): Whether to perform the precomputation steps. Defaults to True.
        """
        self.inputs = [] if inputs is None else inputs
        self.outputs = [] if outputs is None else outputs
        self.stages: List[PipelineStep] = list()
        self.stage_configs = list()
        path = output_path
        for is_last_stage, stage in signal_last(stages):
            requires_saving = output_path is not None and (
                save_intermediate or is_last_stage
            )
            name, config = list(stage.items())[0]
            stage_class = dynamic_import_from(
                f"histocartography.{name}", config.pop("class")
            )
            pipeline_stage = partial(
                stage_class,
                save_path=path if requires_saving else None,
                precompute=False,
                **config.pop("params", {}),
            )
            self.stages.append(pipeline_stage())
            self.stage_configs.append(config)

            if requires_saving:
                assert (
                    self.stages[-1].save_path is not None
                ), f"Cannot update nested path if no save path is defined"
                path = str(self.stages[-1].output_dir)
        self.final_path = path
        if precompute:
            self.precompute(save_intermediate)

    def precompute(self, save_intermediate: bool) -> None:
        """Run the precomputation step of the pipeline.

        Args:
            save_intermediate (bool): Whether to save intermediate outputs
        """
        link_path: Optional[str]
        precompute_path: Optional[str]
        if self.final_path is not None:
            link_path = self.final_path
            if save_intermediate:
                precompute_path = None
            else:
                precompute_path = self.final_path
        else:
            link_path = None
            precompute_path = None

        for stage in self.stages:
            stage.precompute(
                link_path=link_path,
                precompute_path=precompute_path)

    def run(
        self, output_name: Optional[str] = None, **inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the preprocessing pipeline for a given name and input parameters and return the specified outputs

        Args:
            output_name (Optional[str], optional): Unique identifier of the datapoint. Defaults to None.

        Returns:
            Dict[str, Any]: Output of the pipeline as defined in the configuration
        """

        # Validate inputs
        assert (
            output_name is None or self.final_path is not None
        ), f"Saving is only possible when output_path has been passed to the constructor."
        for input_name in self.inputs:
            assert input_name in inputs, f"{input_name} not found in keyword arguments"

        # Compute pipelines steps
        variables = deepcopy(inputs)
        for stage, config in zip(self.stages, self.stage_configs):
            step_input = [variables[k] for k in config["inputs"]]
            step_output = stage.process(*step_input, output_name=output_name)
            if not isinstance(step_output, tuple):
                step_output = tuple([step_output])
            assert len(step_output) == len(config.get("outputs", [])), (
                f"Number of outputs in config mismatches actual number of outputs in {stage.__class__.__name__}"
                f"Got {len(step_output)} outputs of type {list(map(type, step_output))},"
                f"but expected {len(config.get('outputs', []))} outputs"
            )
            for key, value in zip(config.get("outputs", []), step_output):
                variables[key] = value

        # Handle output
        for output_name in self.outputs:
            assert (
                output_name in variables
            ), f"{output_name} should be returned, but was never computed"
        return {k: variables[k] for k in self.outputs}


class BatchPipelineRunner:
    def __init__(
        self,
        pipeline_config: Dict[str, Any],
        save_path: Optional[str],
        save_intermediate: bool = False,
    ) -> None:
        """Run Helper that runs the pipeline for multiple inputs with multiprocessing support

        Args:
            pipeline_config (Dict[str, Any]): Configuration of the pipeline
            save_path (Optional[str]): Path to save the outputs to
            save_intermediate (bool, optional): Whether to save intermediate outputs. Defaults to False.
        """
        self.pipeline_config = pipeline_config
        self.save_path = save_path
        self.save_intermediate = save_intermediate

    def _build_pipeline_runner(self) -> PipelineRunner:
        """Builds and returns a PipelineRunner with the correct configuration

        Returns:
            PipelineRunner: Runner object
        """
        config = deepcopy(self.pipeline_config)
        return PipelineRunner(
            output_path=self.save_path,
            save_intermediate=self.save_intermediate,
            precompute=False,
            **config,
        )

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
        pipeline.run(output_name=name, **row)

    def link_output(self, link_directory: str) -> None:
        """Creates a symlink between the output directory of the pipeline and the provided path.
           Overwrites link if it already exists.

        Args:
            link_directory (str): Path to link the output directory to
        """
        tmp_pipeline = self._build_pipeline_runner()
        final_path = tmp_pipeline.final_path
        assert (
            final_path is not None
        ), "Cannot link output if the pipeline is setup not to save outputs"
        if os.path.exists(link_directory):
            if os.path.islink(link_directory):
                logging.critical("Link already exists: overwriting...")
                os.remove(link_directory)
            else:
                logging.critical(
                    "Link path already exists, but it is something else than a link. Ignoring..."
                )
                return
        os.symlink(final_path, link_directory, target_is_directory=True)
        logging.info(f"Created symlink: {link_directory} -> {final_path}")

    def precompute(self) -> None:
        """Precompute all necessary information for all stages"""
        tmp_runner = self._build_pipeline_runner()
        tmp_runner.precompute(self.save_intermediate)

    def run(
        self, metadata: pd.DataFrame, cores: int = 1, return_out: bool = False
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Runs the pipeline for the provided metadata dataframe and a specified
           number of cores for multiprocessing.
           Does not support saving of outputs

        Args:
            metadata (pd.DataFrame): Dataframe with the columns as defined in the config inputs
            cores (int, optional): Number of cores to use for multiprocessing. Defaults to 1.
            return_out (bool, optional): If the method should also return the output batch data.
                If True, make sure you have enough memory. Only supported
                for single-core processing. Default to False.

        Returns:
            batched_out (Optional[Dict[str, Dict[str, Any]]]): If return_out is True, returns the processed output.
                Otherwise returns None
        """
        assert not (
            return_out and cores > 1
        ), "Option to return output only supported with single-core processing."

        self.precompute()
        if cores == 1:
            batched_out = dict()
            pipeline = self._build_pipeline_runner()
            for name, row in tqdm(
                metadata.iterrows(), total=len(metadata), file=sys.stdout
            ):
                out = pipeline.run(output_name=name, **row)
                if return_out:
                    batched_out[name] = out
            if return_out:
                return batched_out
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
        return None
