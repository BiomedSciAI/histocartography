import copy
import logging
import shutil
from dataclasses import dataclass
from functools import reduce
from itertools import product
from pathlib import Path
from typing import Any, Iterable, List, Union

import yaml

BASE = Path("config") / "default.yml"
PATH = "."


def get_lsf(
    config_name,
    queue="prod.med",
    cores=5,
    gpus=0,
    log_dir="/dataP/anv/logs",
    log_name="preprocessing",
    main_file_name="preprocess",
    nosave=False,
    subsample=None,
    disable_multithreading=False,
    extra_line="",
):
    return (
        f"#!/bin/bash\n\n"
        f"#BSUB -q {queue}\n"
        f"#BSUB -n {cores}\n"
        f"{f'#BSUB -R rusage[ngpus_excl_p={gpus}]' if gpus != 0 else ''}\n"
        f'#BSUB -R "span[hosts=1]"\n'
        f"module load Miniconda3\n"
        f"source activate histocartography\n\n"
        f'#BSUB -J "{log_dir}/{log_name}"\n'
        f'#BSUB -o "{log_dir}/{log_name}"\n'
        f'#BSUB -e "{log_dir}/{log_name}.stderr"\n\n'
        f"{extra_line}"
        f'export PYTHONPATH="$PWD/../../:{{$PYTHONPATH}}"\n'
        f"which python\n"
        f"{'OMP_NUM_THREADS=1 ' if disable_multithreading else ''}"
        f"python {main_file_name}.py "
        f"--config {{PATH}}/{config_name}.yml "
        f"{'--nosave ' if nosave else ''}"
        f"{f'--subsample {subsample}' if subsample is not None else ''}"
        f"\n"
    )


@dataclass
class ParameterList:
    path: List[str]
    value: List[Any]


@dataclass
class Parameter:
    path: List[str]
    value: Any


class Experiment:
    def __init__(
        self,
        name,
        cores=1,
        core_multiplier=6,
        gpus=1,
        subsample=None,
        main_file="train",
        queue="prod.med",
        disable_multithreading=False,
        no_save=False,
        base=BASE,
        path=PATH,
    ) -> None:
        self.name = name
        self.cores = cores
        self.core_mutliplier = core_multiplier
        self.gpus = gpus
        self.subsample = subsample
        self.queue = queue
        self.disable_multithreading = disable_multithreading
        self.no_save = no_save
        self.main_file = main_file

        self.target_directory = Path(path) / self.name
        if not self.target_directory.exists():
            self.target_directory.mkdir()
        else:
            shutil.rmtree(self.target_directory)
            self.target_directory.mkdir()
        self.base = base

    @staticmethod
    def _path_exists(config, path):
        _element = config
        for key in path:
            try:
                _element = _element[key]
            except KeyError:
                return False
        return True

    @staticmethod
    def get(obj, key):
        if isinstance(key, int):
            return obj[key]
        else:
            return obj.get(key)

    @staticmethod
    def _update_config(config, path, value):
        if len(path) > 0:
            if not Experiment._path_exists(config, path):
                logging.warning(
                    f"Config path {path} does not exist. This might be an error"
                )
            try:
                reduce(Experiment.get, path[:-1], config).update({path[-1]: value})
            except AttributeError as e:
                print(f"Could not update {path[-1]}: {value} on path {path[:-1]}")
                pass

    @staticmethod
    def unpack(parameters: Iterable[ParameterList]):
        unpacked = list()
        for parameter in parameters:
            parameter_list = list()
            for parameter_value in parameter.value:
                parameter_list.append(Parameter(parameter.path, parameter_value))
            unpacked.append(parameter_list)
        return unpacked

    @staticmethod
    def grid_product(grid: Iterable[ParameterList]):
        return product(*Experiment.unpack(grid))

    @staticmethod
    def zip(*parameters):
        return zip(*Experiment.unpack(parameters))

    def create_job(self, job_id, config):
        global PATH
        # Generate lsf file
        lsf_content = get_lsf(
            config_name=f"job{job_id}",
            queue=self.queue,
            cores=self.cores,
            gpus=self.gpus,
            log_name=f"{self.name}{job_id}",
            nosave=self.no_save,
            subsample=self.subsample,
            disable_multithreading=self.disable_multithreading,
            main_file_name=self.main_file,
        )

        # Write files
        with open(self.target_directory / f"job{job_id}.lsf", "w") as file:
            file.write(lsf_content)
        with open(self.target_directory / f"job{job_id}.yml", "w") as file:
            yaml.dump(config, file)

    def generate(
        self,
        fixed: Iterable[ParameterList] = (),
        sequential: Union[
            Iterable[ParameterList], Iterable[Iterable[ParameterList]]
        ] = (ParameterList(list(), [None]),),
        grid: Iterable[ParameterList] = (),
        repetitions: int = 1,
    ):
        with open(self.base) as file:
            config: dict = yaml.load(file, Loader=yaml.FullLoader)

        for parameter in fixed:
            self._update_config(config, parameter.path, parameter.value)

        job_id = 0
        for _ in range(repetitions):
            for parameter_combo in sequential:
                if not isinstance(parameter_combo, Iterable):
                    parameter_combo = (parameter_combo,)
                for parameters in self.zip(*parameter_combo):
                    sequential_config = copy.deepcopy(config)
                    for parameter in parameters:
                        self._update_config(
                            sequential_config, parameter.path, parameter.value
                        )
                    if grid:
                        for grid_parameters in self.grid_product(grid):
                            job_config = copy.deepcopy(sequential_config)
                            for grid_parameter in grid_parameters:
                                self._update_config(
                                    job_config,
                                    grid_parameter.path,
                                    grid_parameter.value,
                                )
                            self.create_job(job_id, job_config)
                            job_id += 1
                    else:
                        self.create_job(job_id, sequential_config)
                        job_id += 1


class PretrainingExperiment(Experiment):
    def __init__(
        self, name, queue="prod.med", cores=3, base="config/pretrain.yml", path=PATH, 
    ) -> None:
        super().__init__(
            "pretraining_" + name,
            cores=cores,
            core_multiplier=6,
            gpus=1,
            subsample=None,
            main_file="pretrain",
            queue=queue,
            disable_multithreading=False,
            no_save=False,
            base=base,
            path=path,
        )
        self.name = name
        self.cores = cores

    def generate(self, fixed: Iterable[ParameterList] = list(), **kwargs):
        super().generate(
            [
                Parameter(
                    ["train", "params", "experiment_tags"],
                    {"grid_search": self.name},
                ),
                Parameter(
                    ["train", "params", "num_workers"],
                    self.cores * 8,
                ),
            ]
            + fixed,
            **kwargs,
        )


class MNISTExperiment(Experiment):
    def __init__(self, name, queue="prod.med", path=PATH) -> None:
        super().__init__(
            "mnist_" + name,
            cores=1,
            core_multiplier=6,
            gpus=1,
            subsample=None,
            main_file="train",
            queue=queue,
            disable_multithreading=False,
            no_save=False,
            base="config/mnist.yml",
            path=path
        )
        self.name = name

    def generate(self, fixed: Iterable[ParameterList] = list(), **kwargs):
        super().generate(
            [
                Parameter(
                    ["train", "params", "experiment_tags"],
                    {"grid_search": self.name},
                ),
                Parameter(
                    ["train", "params", "num_workers"],
                    8,
                ),
            ]
            + fixed,
            **kwargs,
        )


class PreprocessingExperiment(Experiment):
    def __init__(self, name, base, **kwargs) -> None:
        super().__init__("preprocessing_" + name, base=base, **kwargs)
        with open(self.base) as file:
            config: dict = yaml.load(file, Loader=yaml.FullLoader)
        self.translator = dict(
            map(
                self._expand,
                list(reversed(list(enumerate(config["pipeline"]["stages"])))),
            )
        )

    def _translate_all(self, arguments):
        for argument in arguments:
            if isinstance(argument, Iterable):
                for subarg in argument:
                    self._translate(subarg)
            else:
                self._translate(argument)

    def _translate(self, el):
        if len(el.path) > 0 and el.path[0] in self.translator:
            el.path = self.translator[el.path[0]] + el.path[1:]

    @staticmethod
    def _expand(x):
        intermediate_key = list(x[1].keys())[0]
        return intermediate_key, ["pipeline", "stages", x[0], intermediate_key]

    def generate(
        self,
        fixed: Iterable[ParameterList] = list(),
        sequential: Union[
            Iterable[ParameterList], Iterable[Iterable[ParameterList]]
        ] = (ParameterList(list(), [None]),),
        grid: Iterable[ParameterList] = (),
    ):
        self._translate_all(fixed)
        self._translate_all(sequential)
        self._translate_all(grid)
        super().generate(
            fixed=fixed,
            sequential=sequential,
            grid=grid,
        )


class GPUPreprocessingExperiment(PreprocessingExperiment):
    def __init__(
        self, name, workers=24, queue="prod.med", base="config/preprocess.yml", **kwargs
    ) -> None:
        self.workers = workers
        super().__init__(
            name,
            cores=max(1, 1 + (workers - 1) // 8),
            gpus=1,
            subsample=None,
            main_file="preprocess",
            queue=queue,
            disable_multithreading=False,
            no_save=False,
            base=base,
            **kwargs,
        )

    def generate(
        self,
        fixed: Iterable[ParameterList] = list(),
        sequential: Union[
            Iterable[ParameterList], Iterable[Iterable[ParameterList]]
        ] = (ParameterList(list(), [None]),),
        grid: Iterable[ParameterList] = (),
    ):
        super().generate(
            fixed=[
                Parameter(["params", "cores"], 1),
                Parameter(
                    ["feature_extraction", "params", "num_workers"], self.workers
                ),
            ]
            + fixed,
            sequential=sequential,
            grid=grid,
        )


class CPUPreprocessingExperiment(PreprocessingExperiment):
    def __init__(
        self, name, cores=4, queue="prod.med", base="config/preprocess.yml", **kwargs
    ) -> None:
        super().__init__(
            name,
            cores=cores,
            core_multiplier=7,
            gpus=0,
            subsample=None,
            main_file="preprocess",
            queue=queue,
            disable_multithreading=True,
            no_save=False,
            base=base,
            **kwargs,
        )
        self.name = name
        self.cores = cores

    def generate(
        self,
        fixed: Iterable[ParameterList] = list(),
        sequential: Union[
            Iterable[ParameterList], Iterable[Iterable[ParameterList]]
        ] = (ParameterList(list(), [None]),),
        grid: Iterable[ParameterList] = (),
    ):
        super().generate(
            fixed=[
                Parameter(["params", "cores"], self.cores * 7 if self.gpus == 0 else 1),
                Parameter(["feature_extraction", "params", "num_workers"], 0),
            ]
            + fixed,
            sequential=sequential,
            grid=grid,
        )


class GraphClassifierExperiment(Experiment):
    def __init__(
        self, name, queue="prod.med", base="config/default.yml", main_file="train", **kwargs
    ) -> None:
        super().__init__(
            "graph_" + name,
            cores=1,
            core_multiplier=6,
            gpus=1,
            subsample=None,
            main_file=main_file,
            queue=queue,
            disable_multithreading=False,
            no_save=False,
            base=base,
            **kwargs
        )
        self.name = name

    def generate(self, fixed: Iterable[ParameterList] = list(), **kwargs):
        super().generate(
            [
                Parameter(
                    ["train", "params", "experiment_tags"],
                    {"grid_search": self.name},
                ),
                Parameter(
                    ["train", "params", "num_workers"],
                    8,
                ),
            ]
            + fixed,
            **kwargs,
        )


class WeaklySupervisedGraphClassificationExperiment(GraphClassifierExperiment):
    def __init__(self, name, queue="prod.med", base="config/default_weak.yml", **kwargs) -> None:
        super().__init__(
            "image_" + name,
            queue=queue,
            base=base,
            main_file="train_weak",
            **kwargs
        )


class StronglySupervisedGraphClassificationExperiment(GraphClassifierExperiment):
    def __init__(
        self, name, queue="prod.med", base="config/default_strong.yml", **kwargs
    ) -> None:
        super().__init__(
            "tissue_" + name,
            queue=queue,
            base=base,
            main_file="train_strong",
            **kwargs, 
        )


class SemiSupervisedGraphClassificationExperiment(GraphClassifierExperiment):
    def __init__(self, name, queue="prod.med", base="config/default_semi.yml", **kwargs) -> None:
        super().__init__(
            "semi_" + name,
            queue=queue,
            base=base,
            main_file="train_semi",
            **kwargs
        )


class CNNTestingExperiment(Experiment):
    def __init__(self, name, queue="prod.short", cores=1, **kwargs) -> None:
        super().__init__(
            "test_cnn_" + name,
            cores=cores,
            core_multiplier=6,
            gpus=1,
            subsample=None,
            main_file="test_cnn",
            queue=queue,
            disable_multithreading=False,
            no_save=False,
            base="config/pretrain.yml",
            **kwargs
        )
        self.name = name
        self.cores = cores

    def generate(self, fixed: Iterable[ParameterList] = list(), **kwargs):
        super().generate(
            [
                Parameter(
                    ["test", "params", "experiment_tags"],
                    {"experiment_name": self.name},
                ),
                Parameter(
                    ["test", "params", "num_workers"],
                    self.cores * 8,
                ),
            ]
            + fixed,
            **kwargs,
        )


class GNNTestingExperiment(Experiment):
    def __init__(
        self, name, queue="prod.short", cores=1, base="config/default.yml", **kwargs
    ) -> None:
        super().__init__(
            "test_gnn_" + name,
            cores=cores,
            core_multiplier=6,
            gpus=1,
            subsample=None,
            main_file="test_gnn",
            queue=queue,
            disable_multithreading=False,
            no_save=False,
            base=base,
            **kwargs
        )
        self.name = name
        self.cores = cores

    def generate(self, fixed: Iterable[ParameterList] = list(), **kwargs):
        super().generate(
            [
                Parameter(
                    ["test", "params", "experiment_tags"],
                    {"experiment_name": self.name},
                )
            ]
            + fixed,
            **kwargs,
        )
