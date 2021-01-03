import argparse
import copy
import logging
import shutil
from dataclasses import dataclass
from functools import reduce
from itertools import product
from pathlib import Path
from typing import Any, Iterable, List, Union

import numpy as np
import yaml

BASE = Path("config") / "default.yml"
PATH = "."


def get_lsf(
    config_name,
    queue="prod.med",
    cores=5,
    gpus=0,
    log_dir="/dataT/anv/logs",
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
        f"module purge\n"
        f"module load Miniconda3\n"
        f"source activate histocartography\n\n"
        f'#BSUB -J "{log_dir}/{log_name}"\n'
        f'#BSUB -o "{log_dir}/{log_name}"\n'
        f'#BSUB -e "{log_dir}/{log_name}.stderr"\n\n'
        f"{extra_line}"
        f'export PYTHONPATH="$PWD/../../:{{$PYTHONPATH}}"\n'
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
        base=None,
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

        self.target_directory = Path(PATH) / self.name
        if not self.target_directory.exists():
            self.target_directory.mkdir()
        else:
            shutil.rmtree(self.target_directory)
            self.target_directory.mkdir()

        if base is None:
            global BASE
            self.base = BASE
        else:
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
                raise e

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
    ):
        with open(self.base) as file:
            config: dict = yaml.load(file, Loader=yaml.FullLoader)

        for parameter in fixed:
            self._update_config(config, parameter.path, parameter.value)

        job_id = 0
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
    def __init__(self, name, queue="prod.med", cores=3) -> None:
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
            base="config/pretrain.yml",
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
    def __init__(self, name, queue="prod.med") -> None:
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
                enumerate(config["pipeline"]["stages"]),
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
        self, name, workers=24, queue="prod.med", base="config/preprocess.yml"
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
        self, name, cores=4, queue="prod.med", base="config/preprocess.yml"
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
            ]
            + fixed,
            sequential=sequential,
            grid=grid,
        )


class GraphClassifierExperiment(Experiment):
    def __init__(self, name, queue="prod.med") -> None:
        super().__init__(
            "graph_" + name,
            cores=1,
            core_multiplier=6,
            gpus=1,
            subsample=None,
            main_file="train",
            queue=queue,
            disable_multithreading=False,
            no_save=False,
            base="config/default.yml",
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


class CNNTestingExperiment(Experiment):
    def __init__(self, name, queue="prod.short", cores=1) -> None:
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
    def __init__(self, name, queue="prod.short", cores=1) -> None:
        super().__init__(
            "test_gnn_" + name,
            cores=cores,
            core_multiplier=6,
            gpus=1,
            subsample=None,
            main_file="test",
            queue=queue,
            disable_multithreading=False,
            no_save=False,
            base="config/default.yml",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="/Users/anv/Documents/experiment_configs"
    )
    parser.add_argument("--base", type=str, default="default.yml")
    args = parser.parse_args()

    PATH = args.path
    BASE = Path("config") / args.base

    # Preprocessing
    CPUPreprocessingExperiment(
        name="superpixels", base="config/superpixel.yml"
    ).generate(
        sequential=[
            ParameterList(
                [
                    "superpixel",
                    "params",
                    "nr_superpixels",
                ],
                [100, 250, 500, 1000, 2000, 4000, 8000],
            )
        ],
    )
    CPUPreprocessingExperiment(name="handcrafted").generate(
        fixed=[
            Parameter(
                ["feature_extraction"],
                {"class": "HandcraftedFeatureExtractor"},
            ),
        ],
        sequential=[
            ParameterList(
                [
                    "superpixel",
                    "params",
                    "nr_superpixels",
                ],
                [500, 1000, 2000],
            )
        ],
    )
    CPUPreprocessingExperiment(name="resnet34",).generate(
        fixed=[
            Parameter(
                ["feature_extraction", "params", "architecture"],
                "resnet34",
            ),
        ],
        sequential=[
            ParameterList(
                [
                    "superpixel",
                    "params",
                    "nr_superpixels",
                ],
                [500, 1000, 2000],
            )
        ],
    )
    CPUPreprocessingExperiment(name="lowres",).generate(
        fixed=[
            Parameter(
                ["feature_extraction", "params", "architecture"],
                "resnet34",
            ),
        ],
        grid=[
            ParameterList(
                [
                    "superpixel",
                    "params",
                    "nr_superpixels",
                ],
                [500, 1000],
            ),
            ParameterList(
                ["feature_extraction", "params", "size"],
                [336, 448, 672],
            ),
        ],
    )
    CPUPreprocessingExperiment(name="v1_few_superpixels",).generate(
        grid=[
            ParameterList(
                [
                    "superpixel",
                    "params",
                    "nr_superpixels",
                ],
                [100, 200, 300, 400, 600, 800],
            ),
            ParameterList(
                ["feature_extraction", "params", "architecture"],
                ["resnet18", "resnet34", "resnet50"],
            ),
        ],
    )
    CPUPreprocessingExperiment(name="v1",).generate(
        grid=[
            ParameterList(
                [
                    "superpixel",
                    "params",
                    "nr_superpixels",
                ],
                [500, 1000],
            ),
            ParameterList(
                ["feature_extraction", "params", "architecture"],
                ["resnet18", "resnet34", "resnet50"],
            ),
        ],
    )
    CPUPreprocessingExperiment(name="v0_few_superpixels",).generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
        ],
        grid=[
            ParameterList(
                [
                    "superpixel",
                    "params",
                    "nr_superpixels",
                ],
                [100, 200, 300, 400, 600, 800],
            ),
            ParameterList(
                ["feature_extraction", "params", "architecture"],
                ["resnet18", "resnet34", "resnet50"],
            ),
        ],
    )
    CPUPreprocessingExperiment(
        name="normalizer_targets",
        base="config/stain_normalizers.yml",
        queue="prod.short",
    ).generate(
        sequential=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                [
                    "ZT111_4_A_1_12",
                    "ZT111_4_A_7_2",
                    "ZT111_4_C_1_12",
                    "ZT199_1_A_2_1",
                    "ZT76_39_B_2_5",
                ],
            )
        ]
    )
    CPUPreprocessingExperiment(
        name="new_superpixels", base="config/superpixel.yml"
    ).generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_A_1_12",
            ),
        ],
        sequential=[
            ParameterList(
                [
                    "superpixel",
                    "params",
                    "nr_superpixels",
                ],
                [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            ),
        ],
    )
    CPUPreprocessingExperiment(name="mobilenet", queue="prod.long").generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_A_1_12",
            ),
            Parameter(["feature_extraction", "params", "architecture"], "mobilenet_v2"),
            Parameter(["feature_extraction", "params", "size"], 672),
        ],
        grid=[
            ParameterList(
                [
                    "superpixel",
                    "params",
                    "nr_superpixels",
                ],
                [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            ),
        ],
    )
    CPUPreprocessingExperiment(name="pretrained", queue="prod.long").generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_A_1_12",
            ),
            Parameter(
                ["feature_extraction", "params", "architecture"],
                "models/485e5ed454714988b70b07ba3231e34d_best_valid_MultiLabelBalancedAccuracy.pth",
            ),
            Parameter(
                ["feature_extraction", "params", "normalizer"],
                {
                    "type": "train",
                    "mean": [0.86489, 0.63272, 0.85928],
                    "std": [0.020820, 0.026320, 0.017309],
                },
            ),
            Parameter(["feature_extraction", "params", "size"], 672),
        ],
        grid=[
            ParameterList(
                [
                    "superpixel",
                    "params",
                    "nr_superpixels",
                ],
                [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            ),
        ],
    )
    CPUPreprocessingExperiment(
        name="pretrained_additional", queue="prod.long"
    ).generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_A_1_12",
            ),
            Parameter(
                ["feature_extraction", "params", "normalizer"],
                {
                    "type": "train",
                    "mean": [0.86489, 0.63272, 0.85928],
                    "std": [0.020820, 0.026320, 0.017309],
                },
            ),
            Parameter(["feature_extraction", "params", "size"], 672),
        ],
        grid=[
            ParameterList(
                ["feature_extraction", "params", "architecture"],
                [
                    "models/d4372ddba84b497fac70a0c5bfc95139_best_valid_MultiLabelBalancedAccuracy.pth",
                    "models/fe014f5eeb9d445a9940fae9422dca73_best_valid_MultiLabelAUCROC.pth",
                ],
            ),
            ParameterList(
                [
                    "superpixel",
                    "params",
                    "nr_superpixels",
                ],
                [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            ),
        ],
    )
    CPUPreprocessingExperiment(
        name="preprare_new_pretraining", queue="prod.long", base="config/superpixel.yml"
    ).generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
        ],
        grid=[
            ParameterList(
                [
                    "superpixel",
                    "params",
                    "nr_superpixels",
                ],
                [300, 600, 900, 1200],
            ),
        ],
    )
    CPUPreprocessingExperiment(name="new_pretraining", queue="prod.long").generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
            Parameter(
                ["feature_extraction", "params", "normalizer"],
                {
                    "type": "train",
                    "mean": [0.86489, 0.63272, 0.85928],
                    "std": [0.020820, 0.026320, 0.017309],
                },
            ),
            Parameter(["feature_extraction", "params", "size"], 672),
        ],
        grid=[
            ParameterList(
                ["feature_extraction", "params", "architecture"],
                [
                    "models/19a9b40d174f40c4b217ddf84eb63e3b_best_valid_MultiLabelBalancedAccuracy.pth",
                    "models/02906fe539444b13a76d39d4a0dfbb6f_best_valid_MultiLabelBalancedAccuracy.pth",
                    "models/c62233eed1574d2ca2d9b8ee51b83ffc_best_valid_MultiLabelBalancedAccuracy.pth",
                ],
            ),
            ParameterList(
                [
                    "superpixel",
                    "params",
                    "nr_superpixels",
                ],
                [300, 600, 900, 1200],
            ),
        ],
    )
    CPUPreprocessingExperiment(name="new_baseline", queue="prod.long").generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
            Parameter(
                ["feature_extraction", "params", "normalizer"],
                {
                    "type": "train",
                    "mean": [0.86489, 0.63272, 0.85928],
                    "std": [0.020820, 0.026320, 0.017309],
                },
            ),
            Parameter(["feature_extraction", "params", "size"], 672),
            Parameter(["feature_extraction", "params", "architecture"], "mobilenet_v2"),
        ],
        grid=[
            ParameterList(
                [
                    "superpixel",
                    "params",
                    "nr_superpixels",
                ],
                [300, 600, 900, 1200],
            ),
        ],
    )
    CPUPreprocessingExperiment(
        name="add_tissue_masks", base="config/stain_normalizers.yml"
    ).generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
        ]
    )
    GPUPreprocessingExperiment(
        name="augmented_new_baseline",
        queue="prod.med",
        base="config/augmented_preprocess.yml",
    ).generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
            Parameter(
                ["feature_extraction", "params", "normalizer"],
                {
                    "type": "train",
                    "mean": [0.86489, 0.63272, 0.85928],
                    "std": [0.020820, 0.026320, 0.017309],
                },
            ),
            Parameter(["feature_extraction", "params", "size"], 672),
            Parameter(["feature_extraction", "params", "architecture"], "mobilenet_v2"),
        ],
        sequential=[
            [
                ParameterList(
                    [
                        "superpixel",
                        "params",
                        "nr_superpixels",
                    ],
                    [300, 600, 900, 1200],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [f"v4_mobilenet_{s}" for s in [300, 600, 900, 1200]],
                ),
            ]
        ],
    )

    # ETH
    GraphClassifierExperiment(name="node_stochasticity").generate(
        sequential=[
            ParameterList(
                ["train", "params", "loss", "node", "params", "drop_probability"],
                [0.99, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1],
            ),
        ],
    )
    GraphClassifierExperiment(name="node_loss_weight").generate(
        sequential=[
            ParameterList(
                ["train", "params", "loss", "node_weight"],
                [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            ),
        ],
    )
    GraphClassifierExperiment(name="batch_sizes").generate(
        sequential=[
            ParameterList(
                ["train", "params", "batch_size"],
                [1, 2, 4, 8, 16, 32, 64],
            ),
        ],
    )
    GraphClassifierExperiment(name="learning_rates").generate(
        sequential=[
            ParameterList(
                ["train", "params", "optimizer", "params", "lr"],
                list(map(float, np.logspace(-2, -6, 15))),
            ),
        ],
    )
    GraphClassifierExperiment(name="nr_superpixels").generate(
        sequential=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    "outputs/v0_100px",
                    "outputs/v0_250px",
                    "outputs/v0_500px",
                    "outputs/v0_1000px",
                    "outputs/v0_2000px",
                    "outputs/v0_4000px",
                ],
            ),
        ],
    )
    GraphClassifierExperiment(name="crop_augmentation").generate(
        sequential=[
            ParameterList(
                ["train", "data", "patch_size"],
                [1900, 2200, 2800, 3100],
            ),
        ],
    )
    GraphClassifierExperiment(name="gnn_parameters").generate(
        grid=[
            ParameterList(
                ["train", "model", "gnn_config", "agg_operator"],
                ["none", "concat", "lstm"],
            ),
            ParameterList(["train", "model", "gnn_config", "n_layers"], [2, 4, 6]),
            ParameterList(["train", "model", "gnn_config", "hidden_dim"], [32, 64]),
        ],
    )
    GraphClassifierExperiment(name="centroid_features").generate(
        sequential=[
            [
                ParameterList(["train", "data", "centroid_features"], ["only"]),
                ParameterList(["train", "model", "gnn_config", "input_dim"], [2]),
            ]
        ],
    )
    GraphClassifierExperiment(name="lr_gradient_norm").generate(
        fixed=[
            Parameter(["train", "params", "clip_gradient_norm"], 5.0),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "optimizer", "params", "lr"],
                list(map(float, np.logspace(-2, -5, 10))),
            ),
        ],
    )
    GraphClassifierExperiment(name="tiny_centroid_networks").generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "only"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 2),
            Parameter(
                ["train", "params", "loss", "node", "params", "drop_probability"], 0.5
            ),
        ],
        sequential=[
            [
                ParameterList(["train", "model", "gnn_config", "n_layers"], [2, 2, 2]),
                ParameterList(
                    ["train", "model", "gnn_config", "hidden_dim"], [16, 8, 4]
                ),
                ParameterList(
                    ["train", "model", "gnn_config", "output_dim"], [16, 8, 4]
                ),
                ParameterList(
                    ["train", "model", "graph_classifier_config", "hidden_dim"],
                    [32, 16, 8],
                ),
            ],
            [
                ParameterList(["train", "model", "gnn_config", "n_layers"], [3, 3, 3]),
                ParameterList(
                    ["train", "model", "gnn_config", "hidden_dim"], [16, 8, 4]
                ),
                ParameterList(
                    ["train", "model", "gnn_config", "output_dim"], [16, 8, 4]
                ),
                ParameterList(
                    ["train", "model", "graph_classifier_config", "hidden_dim"],
                    [48, 24, 12],
                ),
            ],
            [
                ParameterList(["train", "model", "gnn_config", "n_layers"], [4, 4, 4]),
                ParameterList(
                    ["train", "model", "gnn_config", "hidden_dim"], [16, 8, 4]
                ),
                ParameterList(
                    ["train", "model", "gnn_config", "output_dim"], [16, 8, 4]
                ),
                ParameterList(
                    ["train", "model", "graph_classifier_config", "hidden_dim"],
                    [64, 32, 16],
                ),
            ],
        ],
    )
    GraphClassifierExperiment(name="small_networks").generate(
        fixed=[
            Parameter(
                ["train", "params", "loss", "node", "params", "drop_probability"], 0.5
            ),
        ],
        sequential=[
            [
                ParameterList(["train", "model", "gnn_config", "n_layers"], [3, 3, 3]),
                ParameterList(
                    ["train", "model", "gnn_config", "hidden_dim"], [32, 16, 8]
                ),
                ParameterList(
                    ["train", "model", "gnn_config", "output_dim"], [32, 16, 8]
                ),
            ],
            [
                ParameterList(["train", "model", "gnn_config", "n_layers"], [4, 4, 4]),
                ParameterList(
                    ["train", "model", "gnn_config", "hidden_dim"], [32, 16, 8]
                ),
                ParameterList(
                    ["train", "model", "gnn_config", "output_dim"], [32, 16, 8]
                ),
            ],
        ],
    )
    GraphClassifierExperiment(name="baseline").generate(
        sequential=[
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers": 2, "hidden_dim": 16}],
            )
        ],
    )
    GraphClassifierExperiment(name="gnn_agg").generate(
        grid=[
            ParameterList(
                ["train", "model", "gnn_config", "neighbor_pooling_type"],
                ["sum", "min", "max"],
            ),
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers": 2, "hidden_dim": 16}],
            ),
        ],
    )
    GraphClassifierExperiment(name="normalize").generate(
        grid=[
            ParameterList(["train", "data", "normalize_features"], [True]),
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers": 2, "hidden_dim": 16}],
            ),
        ],
    )
    GraphClassifierExperiment(name="fold_sanity_check").generate(
        sequential=[
            [
                ParameterList(
                    ["train", "data", "training_slides"],
                    [[76, 199, 204], [111, 76, 199], [111, 76, 204], [111, 199, 204]],
                ),
                ParameterList(
                    ["train", "data", "validation_slides"], [[111], [204], [199], [76]]
                ),
            ]
        ],
    )
    GraphClassifierExperiment(name="handcrafted_features").generate(
        fixed=[
            Parameter(
                ["train", "data", "normalize_features"],
                True,
            ),
            Parameter(["train", "model", "gnn_config", "input_dim"], 59),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                ["outputs/hc_500px", "outputs/hc_1000px", "outputs/hc_2000px"],
            ),
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers": 2, "hidden_dim": 16}],
            ),
        ],
    )
    GraphClassifierExperiment(name="resnet34").generate(
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    "outputs/resnet34_500px",
                    "outputs/resnet34_1000px",
                    "outputs/resnet34_2000px",
                ],
            ),
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers": 2, "hidden_dim": 16}],
            ),
        ],
    )
    GraphClassifierExperiment(name="downsampled_1.5x").generate(
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                ["outputs/resnet34_500px_1.5x", "outputs/resnet34_1000px_1.5x"],
            ),
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers": 2, "hidden_dim": 16}],
            ),
        ],
    )
    GraphClassifierExperiment(name="downsampled_2x").generate(
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                ["outputs/resnet34_500px_2x", "outputs/resnet34_1000px_2x"],
            ),
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers": 2, "hidden_dim": 16}],
            ),
        ],
    )
    GraphClassifierExperiment(name="downsampled_3x").generate(
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                ["outputs/resnet34_500px_3x", "outputs/resnet34_1000px_3x"],
            ),
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers": 2, "hidden_dim": 16}],
            ),
        ],
    )
    GraphClassifierExperiment(name="mixed_slides").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 4000),
            ParameterList(["train", "model", "node_classifier_config"], None),
        ],
        sequential=[
            [
                ParameterList(["train", "data", "train_fraction"], 3 * [0.3]),
                ParameterList(["train", "params", "seed"], [0, 1, 2]),
            ]
        ],
    )
    GraphClassifierExperiment(name="mixed_slides_node_only").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 4000),
        ],
        sequential=[
            [
                ParameterList(["train", "data", "train_fraction"], 3 * [0.3]),
                ParameterList(["train", "params", "seed"], [0, 1, 2]),
            ]
        ],
    )
    GraphClassifierExperiment(name="v1_mixed_slides_500px").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 4000),
            ParameterList(["train", "model", "node_classifier_config"], None),
            Parameter(["train", "data", "graph_directory"], "outputs/v1_500px"),
        ],
        sequential=[
            [
                ParameterList(["train", "data", "train_fraction"], 3 * [0.3]),
                ParameterList(["train", "params", "seed"], [0, 1, 2]),
            ]
        ],
    )
    GraphClassifierExperiment(name="v1_mixed_slides_node_only_500px").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 4000),
            Parameter(["train", "data", "graph_directory"], "outputs/v1_500px"),
        ],
        sequential=[
            [
                ParameterList(["train", "data", "train_fraction"], 3 * [0.3]),
                ParameterList(["train", "params", "seed"], [0, 1, 2]),
            ]
        ],
    )
    GraphClassifierExperiment(name="v0_mixed_slides_500px").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 4000),
            Parameter(["train", "data", "graph_directory"], "outputs/v0_500px"),
            Parameter(["train", "data", "train_fraction"], 0.3),
            ParameterList(["train", "params", "seed"], 0),
        ],
        grid=[
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers": 2, "hidden_dim": 16}],
            )
        ],
    )
    GraphClassifierExperiment(name="v1_node_classifier_500px").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(
                ["train", "data", "graph_directory"], "outputs/v1_500px_resnet18"
            ),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "loss", "node", "params", "drop_probability"],
                [0.8, 0.6, 0.4, 0.2, 0.0],
            ),
        ],
    )
    GraphClassifierExperiment(name="v1_node_classifier_1000px").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(
                ["train", "data", "graph_directory"], "outputs/v1_1000px_resnet18"
            ),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "loss", "node", "params", "drop_probability"],
                [0.8, 0.6, 0.4, 0.2, 0.0],
            ),
        ],
    )
    GraphClassifierExperiment(name="v1_node_classifier_2000px").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(
                ["train", "data", "graph_directory"], "outputs/v1_2000px_resnet34"
            ),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "loss", "node", "params", "drop_probability"],
                [0.8, 0.6, 0.4, 0.2, 0.0],
            ),
        ],
    )
    GraphClassifierExperiment(name="v1_node_small_network_1000px").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(
                ["train", "data", "graph_directory"], "outputs/v1_1000px_resnet18"
            ),
            Parameter(["train", "params", "validation_frequency"], 20),
        ],
        sequential=[
            [
                ParameterList(["train", "model", "gnn_config", "n_layers"], [3, 3, 3]),
                ParameterList(["train", "model", "gnn_config", "hidden_dim"], [16, 8]),
                ParameterList(["train", "model", "gnn_config", "output_dim"], [16, 8]),
            ],
            [
                ParameterList(["train", "model", "gnn_config", "n_layers"], [4, 4, 4]),
                ParameterList(["train", "model", "gnn_config", "hidden_dim"], [16, 8]),
                ParameterList(["train", "model", "gnn_config", "output_dim"], [16, 8]),
            ],
            [
                ParameterList(
                    ["train", "model", "node_classifier_config", "hidden_dim"],
                    [16, 8, 8],
                ),
                ParameterList(
                    ["train", "model", "node_classifier_config", "n_layers"], [2, 2, 3]
                ),
            ],
        ],
        grid=[
            ParameterList(
                ["train", "model", "gnn_config", "neighbor_pooling_type"],
                ["sum", "min", "max", "mean"],
            ),
        ],
    )
    GraphClassifierExperiment(name="v1_few_spx_small_nw_resnet18").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "validation_frequency"], 20),
        ],
        sequential=[
            [
                ParameterList(["train", "model", "gnn_config", "n_layers"], [3, 3, 3]),
                ParameterList(
                    ["train", "model", "gnn_config", "hidden_dim"], [32, 16, 8]
                ),
                ParameterList(
                    ["train", "model", "gnn_config", "output_dim"], [32, 16, 8]
                ),
                ParameterList(
                    ["train", "model", "node_classifier_config", "hidden_dim"],
                    [16, 8, 4],
                ),
            ]
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v1_{s}px_resnet{n}"
                    for n in [18]
                    for s in [100, 200, 300, 400, 600, 800]
                ],
            ),
        ],
    )
    GraphClassifierExperiment(name="v1_few_spx_small_nw_resnet34").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "validation_frequency"], 20),
        ],
        sequential=[
            [
                ParameterList(["train", "model", "gnn_config", "n_layers"], [3, 3, 3]),
                ParameterList(
                    ["train", "model", "gnn_config", "hidden_dim"], [32, 16, 8]
                ),
                ParameterList(
                    ["train", "model", "gnn_config", "output_dim"], [32, 16, 8]
                ),
                ParameterList(
                    ["train", "model", "node_classifier_config", "hidden_dim"],
                    [16, 8, 4],
                ),
            ]
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v1_{s}px_resnet{n}"
                    for n in [34]
                    for s in [100, 200, 300, 400, 600, 800]
                ],
            ),
        ],
    )
    GraphClassifierExperiment(name="v1_few_spx_small_nw_resnet50").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "validation_frequency"], 20),
        ],
        sequential=[
            [
                ParameterList(["train", "model", "gnn_config", "n_layers"], [3, 3, 3]),
                ParameterList(
                    ["train", "model", "gnn_config", "hidden_dim"], [32, 16, 8]
                ),
                ParameterList(
                    ["train", "model", "gnn_config", "output_dim"], [32, 16, 8]
                ),
                ParameterList(
                    ["train", "model", "node_classifier_config", "hidden_dim"],
                    [16, 8, 4],
                ),
            ]
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v1_{s}px_resnet{n}"
                    for n in [50]
                    for s in [100, 200, 300, 400, 600, 800]
                ],
            ),
        ],
    )
    GraphClassifierExperiment(name="v0_check2_2000px").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config", "hidden_dim"], 64),
            Parameter(["train", "params", "validation_frequency"], 20),
            Parameter(["train", "model", "gnn_config", "n_layers"], 4),
            Parameter(["train", "model", "gnn_config", "hidden_dim"], 64),
            Parameter(["train", "model", "gnn_config", "output_dim"], 64),
            Parameter(["train", "model", "node_classifier_config", "hidden_dim"], 32),
            Parameter(["train", "params", "loss", "node_weight"], 1.0),
        ],
        sequential=[
            [
                ParameterList(["train", "params", "seed"], [None, 1802989081]),
            ]
        ],
    )
    GraphClassifierExperiment(name="v0_few_spx_small_nw").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "validation_frequency"], 20),
        ],
        sequential=[
            [
                ParameterList(["train", "model", "gnn_config", "n_layers"], [3, 3, 3]),
                ParameterList(
                    ["train", "model", "gnn_config", "hidden_dim"], [32, 16, 8]
                ),
                ParameterList(
                    ["train", "model", "gnn_config", "output_dim"], [32, 16, 8]
                ),
                ParameterList(
                    ["train", "model", "node_classifier_config", "hidden_dim"],
                    [16, 8, 4],
                ),
            ]
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [f"outputs/v0_{s}px" for s in [250, 500, 1000]],
            ),
        ],
    )
    GraphClassifierExperiment(name="v2_learning_rates_fixed_seed").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "seed"], 42),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "optimizer", "params", "lr"],
                list(map(float, np.logspace(-2, -6, 10))),
            ),
        ],
    )
    GraphClassifierExperiment(name="v2_mobilenet").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 5000),
        ],
        sequential=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v2_{x}_mobilenet_v2"
                    for x in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v2_patch_size").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 5000),
            Parameter(
                ["train", "data", "graph_directory"], "outputs/v2_500_mobilenet_v2"
            ),
        ],
        sequential=[
            ParameterList(
                ["train", "data", "patch_size"],
                [1500, 2000, 2400, 2600, 2800, 3000, None],
            )
        ],
    )
    GraphClassifierExperiment(name="v2_batch_size").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 5000),
            Parameter(
                ["train", "data", "graph_directory"], "outputs/v2_500_mobilenet_v2"
            ),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "batch_size"],
                [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256],
            )
        ],
    )
    GraphClassifierExperiment(name="v2_pretrained").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 5000),
        ],
        sequential=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v2_{x}_Local(models_485e5ed454714988b70b07ba3231e34d_best_valid_MultiLabelBalancedAccuracy.pth)"
                    for x in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v2_pretrained_additional_1").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
        ],
        sequential=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v2_{x}_Local(models_d4372ddba84b497fac70a0c5bfc95139_best_valid_MultiLabelBalancedAccuracy.pth)"
                    for x in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v2_pretrained_additional_2").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
        ],
        sequential=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v2_{x}_Local(models_fe014f5eeb9d445a9940fae9422dca73_best_valid_MultiLabelAUCROC.pth)"
                    for x in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v2_shared_head").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 500),
        ],
        grid=[
            ParameterList(
                ["train", "model", "node_classifier_config", "seperate_heads"],
                [True, False],
            ),
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    "outputs/v2_500_Local(models_485e5ed454714988b70b07ba3231e34d_best_valid_MultiLabelBalancedAccuracy.pth)",
                    "outputs/v2_500_mobilenet_v2",
                ],
            ),
        ],
    )
    GraphClassifierExperiment(name="v2_anti_overfitting").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], False
            ),
        ],
        grid=[
            ParameterList(
                ["train", "model", "node_classifier_config", "input_dropout"],
                [0.5, 0.2, 0.7, 0.0],
            ),
            ParameterList(
                ["train", "model", "node_classifier_config", "n_layers"],
                [1, 2, 3],
            ),
            ParameterList(
                ["train", "model", "node_classifier_config", "hidden_dim"], [16, 32, 64]
            ),
        ],
    )
    GraphClassifierExperiment(name="v2_anti_overfitting_pretrained").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], False
            ),
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v2_500_Local(models_485e5ed454714988b70b07ba3231e34d_best_valid_MultiLabelBalancedAccuracy.pth)",
            ),
        ],
        grid=[
            ParameterList(
                ["train", "model", "node_classifier_config", "input_dropout"],
                [0.5, 0.2, 0.7, 0.0],
            ),
            ParameterList(
                ["train", "model", "node_classifier_config", "n_layers"],
                [1, 2, 3],
            ),
            ParameterList(
                ["train", "model", "node_classifier_config", "hidden_dim"], [16, 32, 64]
            ),
        ],
    )
    GraphClassifierExperiment(name="v2_gnn_reg").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], False
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "input_dropout"], 0.5
            ),
        ],
        grid=[
            ParameterList(
                ["train", "model", "gnn_config", "dropout"],
                [0.5, 0.2, 0.7, 0.0],
            ),
            ParameterList(
                ["train", "model", "gnn_config", "n_layers"],
                [2, 4, 6],
            ),
        ],
    )
    GraphClassifierExperiment(name="v2_gnn_reg_pretrained").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], False
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "input_dropout"], 0.5
            ),
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v2_500_Local(models_485e5ed454714988b70b07ba3231e34d_best_valid_MultiLabelBalancedAccuracy.pth)",
            ),
        ],
        grid=[
            ParameterList(
                ["train", "model", "gnn_config", "dropout"],
                [0.5, 0.2, 0.7, 0.0],
            ),
            ParameterList(
                ["train", "model", "gnn_config", "n_layers"],
                [2, 4, 6],
            ),
        ],
    )
    GraphClassifierExperiment(name="v3_baseline").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], False
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "input_dropout"], 0.5
            ),
        ]
    )
    GraphClassifierExperiment(name="v3_gnn_deep").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], False
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "input_dropout"], 0.5
            ),
            Parameter(
                ["train", "model", "gnn_config", "dropout"],
                0.5,
            ),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                6,
            ),
            Parameter(
                ["train", "model", "gnn_config", "hidden_dim"],
                32,
            ),
            Parameter(
                ["train", "model", "gnn_config", "output_dim"],
                32,
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "n_layers"],
                2,
            ),
            Parameter(["train", "model", "node_classifier_config", "hidden_dim"], 32),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v3_{x}_{y}"
                    for x in [300, 600, 900, 1200]
                    for y in [
                        "02906fe539444b13a76d39d4a0dfbb6f",
                        "19a9b40d174f40c4b217ddf84eb63e3b",
                        "c62233eed1574d2ca2d9b8ee51b83ffc",
                    ]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v3_gnn_wide").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], False
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "input_dropout"], 0.5
            ),
            Parameter(
                ["train", "model", "gnn_config", "dropout"],
                0.5,
            ),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                4,
            ),
            Parameter(
                ["train", "model", "gnn_config", "hidden_dim"],
                64,
            ),
            Parameter(
                ["train", "model", "gnn_config", "output_dim"],
                64,
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "n_layers"],
                2,
            ),
            Parameter(["train", "model", "node_classifier_config", "hidden_dim"], 64),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v3_{x}_{y}"
                    for x in [300, 600, 900, 1200]
                    for y in [
                        "02906fe539444b13a76d39d4a0dfbb6f",
                        "19a9b40d174f40c4b217ddf84eb63e3b",
                        "c62233eed1574d2ca2d9b8ee51b83ffc",
                    ]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v3_gnn_small").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], False
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "input_dropout"], 0.5
            ),
            Parameter(
                ["train", "model", "gnn_config", "dropout"],
                0.5,
            ),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                2,
            ),
            Parameter(
                ["train", "model", "gnn_config", "hidden_dim"],
                16,
            ),
            Parameter(
                ["train", "model", "gnn_config", "output_dim"],
                16,
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "n_layers"],
                2,
            ),
            Parameter(["train", "model", "node_classifier_config", "hidden_dim"], 16),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v3_{x}_{y}"
                    for x in [300, 600, 900, 1200]
                    for y in [
                        "02906fe539444b13a76d39d4a0dfbb6f",
                        "19a9b40d174f40c4b217ddf84eb63e3b",
                        "c62233eed1574d2ca2d9b8ee51b83ffc",
                    ]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v3_baseline_gnn_deep").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], False
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "input_dropout"], 0.5
            ),
            Parameter(
                ["train", "model", "gnn_config", "dropout"],
                0.5,
            ),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                6,
            ),
            Parameter(
                ["train", "model", "gnn_config", "hidden_dim"],
                32,
            ),
            Parameter(
                ["train", "model", "gnn_config", "output_dim"],
                32,
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "n_layers"],
                2,
            ),
            Parameter(["train", "model", "node_classifier_config", "hidden_dim"], 32),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v3_{x}_{y}"
                    for x in [300, 600, 900, 1200]
                    for y in ["mobilenet_v2"]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v3_baseline_gnn_wide").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], False
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "input_dropout"], 0.5
            ),
            Parameter(
                ["train", "model", "gnn_config", "dropout"],
                0.5,
            ),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                4,
            ),
            Parameter(
                ["train", "model", "gnn_config", "hidden_dim"],
                64,
            ),
            Parameter(
                ["train", "model", "gnn_config", "output_dim"],
                64,
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "n_layers"],
                2,
            ),
            Parameter(["train", "model", "node_classifier_config", "hidden_dim"], 64),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v3_{x}_{y}"
                    for x in [300, 600, 900, 1200]
                    for y in [
                        "mobilenet_v2",
                    ]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v3_baseline_gnn_small").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], False
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "input_dropout"], 0.5
            ),
            Parameter(
                ["train", "model", "gnn_config", "dropout"],
                0.5,
            ),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                2,
            ),
            Parameter(
                ["train", "model", "gnn_config", "hidden_dim"],
                16,
            ),
            Parameter(
                ["train", "model", "gnn_config", "output_dim"],
                16,
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "n_layers"],
                2,
            ),
            Parameter(["train", "model", "node_classifier_config", "hidden_dim"], 16),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v3_{x}_{y}"
                    for x in [300, 600, 900, 1200]
                    for y in [
                        "mobilenet_v2",
                    ]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v3_gnn_deep").generate(
        fixed=[
            Parameter(["train", "model", "graph_classifier_config"], None),
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], False
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "input_dropout"], 0.5
            ),
            Parameter(
                ["train", "model", "gnn_config", "dropout"],
                0.5,
            ),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                6,
            ),
            Parameter(
                ["train", "model", "gnn_config", "hidden_dim"],
                32,
            ),
            Parameter(
                ["train", "model", "gnn_config", "output_dim"],
                32,
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "n_layers"],
                2,
            ),
            Parameter(["train", "model", "node_classifier_config", "hidden_dim"], 32),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v3_{x}_{y}"
                    for x in [300, 600, 900, 1200]
                    for y in [
                        "02906fe539444b13a76d39d4a0dfbb6f",
                        "19a9b40d174f40c4b217ddf84eb63e3b",
                        "c62233eed1574d2ca2d9b8ee51b83ffc",
                    ]
                ],
            )
        ],
    )

    # MNIST
    MNISTExperiment(name="batch_sizes").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 500),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "batch_size"],
                [8, 16, 32, 64, 128],
            ),
        ],
    )
    MNISTExperiment(name="learning_rates").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 500),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "optimizer", "params", "lr"],
                list(map(float, np.logspace(-3, -6, 10)))[5:],
            ),
        ],
    )
    MNISTExperiment(name="centroid_features").generate(
        sequential=[
            [
                ParameterList(["train", "data", "centroid_features"], ["no", "only"]),
                ParameterList(["train", "model", "gnn_config", "input_dim"], [57, 2]),
            ],
        ],
    ),
    MNISTExperiment(name="no_normalizer").generate(
        sequential=[
            ParameterList(["train", "data", "normalize_features"], [True, False]),
        ],
    ),
    MNISTExperiment(name="gnn_config").generate(
        sequential=[
            [
                ParameterList(
                    ["train", "model", "graph_classifier_config", "n_layers"],
                    [2, 2, 3, 3],
                ),
                ParameterList(
                    ["train", "model", "graph_classifier_config", "hidden_dim"],
                    [32, 48, 32, 48],
                ),
            ],
            [
                ParameterList(["train", "model", "gnn_config", "n_layers"], [2, 2, 2]),
                ParameterList(
                    ["train", "model", "gnn_config", "hidden_dim"], [32, 16, 8]
                ),
                ParameterList(
                    ["train", "model", "gnn_config", "output_dim"], [32, 16, 8]
                ),
            ],
            [
                ParameterList(["train", "model", "gnn_config", "n_layers"], [3, 3, 3]),
                ParameterList(
                    ["train", "model", "gnn_config", "hidden_dim"], [32, 16, 8]
                ),
                ParameterList(
                    ["train", "model", "gnn_config", "output_dim"], [32, 16, 8]
                ),
            ],
            [
                ParameterList(["train", "model", "gnn_config", "n_layers"], [4, 4, 4]),
                ParameterList(
                    ["train", "model", "gnn_config", "hidden_dim"], [32, 16, 8]
                ),
                ParameterList(
                    ["train", "model", "gnn_config", "output_dim"], [32, 16, 8]
                ),
            ],
        ],
    )
    MNISTExperiment(name="aggregation").generate(
        sequential=[
            ParameterList(
                ["train", "model", "gnn_config", "neighbor_pooling_type"],
                ["mean", "sum", "min", "max"],
            ),
        ],
    )

    # Pretraining
    PretrainingExperiment(name="learning_rates").generate(
        sequential=[
            ParameterList(
                ["train", "params", "optimizer", "params", "lr"],
                list(map(float, np.logspace(-4, -6, 10))),
            ),
        ]
    )
    PretrainingExperiment(name="batch_sizes").generate(
        sequential=[
            ParameterList(
                ["train", "params", "batch_size"],
                [48, 64, 96, 128, 192],
            ),
        ]
    )
    PretrainingExperiment(name="basic_architectures").generate(
        grid=[
            ParameterList(
                ["train", "model", "architecture"],
                ["resnet18", "resnet34"],
            ),
            ParameterList(
                ["train", "model", "pretrained"],
                [True, False],
            ),
        ]
    )
    PretrainingExperiment(name="scratch_augmentations").generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.001,
            ),
            Parameter(
                ["train", "model", "pretrained"],
                False,
            ),
        ],
        grid=[
            ParameterList(
                ["train", "data", "augmentations"],
                [
                    {k: v for d in l for k, v in d.items()}
                    for l in list(
                        product(
                            [
                                {"rotation": {"degrees": 180, "crop": 224}},
                                {},
                            ],
                            [{"flip": None}, {}],
                            [
                                {
                                    "color_jitter": {
                                        "saturation": 0.5,
                                        "contrast": 0.3,
                                        "brightness": 0.3,
                                        "hue": 0.1,
                                    }
                                },
                                {"color_jitter": {"contrast": 0.3, "brightness": 0.3}},
                                {"color_jitter": {"saturation": 0.5, "hue": 0.1}},
                                {},
                            ],
                        )
                    )
                ],
            ),
            ParameterList(
                ["train", "data", "normalizer"],
                [
                    None,
                    {
                        "type": "train",
                        "mean": [0.86489, 0.63272, 0.85928],
                        "std": [0.020820, 0.026320, 0.017309],
                    },
                ],
            ),
        ],
    )
    PretrainingExperiment(name="fine_tune_architectures").generate(
        fixed=[
            Parameter(["train", "params", "optimizer", "class"], "SGD"),
            Parameter(
                ["train", "params", "optimizer", "params"],
                {"lr": 0.0001, "momentum": 0.9, "nesterov": True},
            ),
            Parameter(
                ["train", "model", "pretrained"],
                True,
            ),
        ],
        sequential=[
            ParameterList(
                ["train", "model", "architecture"],
                [
                    "mobilenet_v2",
                    "resnet18",
                    "mnasnet0_5",
                    "wide_resnet50_2",
                    "resnet34",
                    "resnet50",
                    "resnext50_32x4d",
                ],
            )
        ],
    )
    PretrainingExperiment(name="scratch_architectures").generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.001,
            ),
            Parameter(
                ["train", "model", "pretrained"],
                False,
            ),
        ],
        sequential=[
            ParameterList(
                ["train", "model", "architecture"],
                [
                    "mobilenet_v2",
                    "resnet18",
                    "mnasnet0_5",
                    "wide_resnet50_2",
                    "resnet34",
                    "resnet50",
                    "resnext50_32x4d",
                ],
            )
        ],
    )
    PretrainingExperiment(name="fine_tune_mixed_architectures").generate(
        fixed=[
            Parameter(["train", "params", "optimizer", "class"], "SGD"),
            Parameter(
                ["train", "params", "optimizer", "params"],
                {"lr": 0.0001, "momentum": 0.9, "nesterov": True},
            ),
            Parameter(
                ["train", "model", "pretrained"],
                True,
            ),
            Parameter(["train", "data", "train_fraction"], 0.8),
            Parameter(["train", "data", "training_slides"], None),
            Parameter(["train", "data", "validation_slides"], None),
        ],
        sequential=[
            ParameterList(
                ["train", "model", "architecture"],
                [
                    "mobilenet_v2",
                    "resnet18",
                    "mnasnet0_5",
                    "wide_resnet50_2",
                    "resnet34",
                    "resnet50",
                    "resnext50_32x4d",
                ],
            )
        ],
    )
    PretrainingExperiment(name="scratch_mixed_architectures").generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.001,
            ),
            Parameter(
                ["train", "model", "pretrained"],
                False,
            ),
            Parameter(["train", "data", "train_fraction"], 0.8),
            Parameter(["train", "data", "training_slides"], None),
            Parameter(["train", "data", "validation_slides"], None),
        ],
        sequential=[
            ParameterList(
                ["train", "model", "architecture"],
                [
                    "mobilenet_v2",
                    "resnet18",
                    "mnasnet0_5",
                    "wide_resnet50_2",
                    "resnet34",
                    "resnet50",
                    "resnext50_32x4d",
                ],
            )
        ],
    )
    PretrainingExperiment(name="dropout_test").generate(
        fixed=[
            Parameter(["train", "params", "optimizer", "class"], "SGD"),
            Parameter(
                ["train", "params", "optimizer", "params"],
                {"lr": 0.0001, "momentum": 0.9, "nesterov": True},
            ),
            Parameter(
                ["train", "model", "pretrained"],
                True,
            ),
            Parameter(["train", "data", "train_fraction"], 0.8),
            Parameter(["train", "data", "training_slides"], None),
            Parameter(["train", "data", "validation_slides"], None),
            Parameter(["train", "model", "architecture"], "mobilenet_v2"),
        ],
        grid=[
            ParameterList(
                ["train", "model", "dropout"], [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
            ),
        ],
    )
    PretrainingExperiment(name="no_spatial_fine_tune_augmentations").generate(
        fixed=[
            Parameter(["train", "params", "optimizer", "class"], "SGD"),
            Parameter(
                ["train", "params", "optimizer", "params"],
                {"lr": 0.0001, "momentum": 0.9, "nesterov": True},
            ),
            Parameter(["train", "data", "train_fraction"], 0.8),
            Parameter(["train", "data", "training_slides"], None),
            Parameter(["train", "data", "validation_slides"], None),
            Parameter(["train", "model", "architecture"], "mobilenet_v2"),
        ],
        grid=[
            ParameterList(
                ["train", "data", "augmentations"],
                [
                    {k: v for d in l for k, v in d.items()}
                    for l in list(
                        product(
                            [
                                {
                                    "rotation": {"degrees": 180, "crop": 224},
                                    "flip": None,
                                },
                            ],
                            [
                                {
                                    "color_jitter": {
                                        "saturation": 0.6,
                                        "contrast": 0.5,
                                        "brightness": 0.5,
                                        "hue": 0.5,
                                    }
                                },
                                {
                                    "color_jitter": {
                                        "saturation": 0.5,
                                        "contrast": 0.4,
                                        "brightness": 0.4,
                                        "hue": 0.3,
                                    }
                                },
                                {
                                    "color_jitter": {
                                        "saturation": 0.5,
                                        "contrast": 0.3,
                                        "brightness": 0.3,
                                        "hue": 0.1,
                                    }
                                },
                                {},
                            ],
                        )
                    )
                ],
            ),
            ParameterList(
                ["train", "data", "normalizer"],
                [
                    None,
                    {
                        "type": "train",
                        "mean": [0.86489, 0.63272, 0.85928],
                        "std": [0.020820, 0.026320, 0.017309],
                    },
                ],
            ),
        ],
    )
    PretrainingExperiment(name="spatial_fine_tune_augmentations").generate(
        fixed=[
            Parameter(["train", "params", "optimizer", "class"], "SGD"),
            Parameter(
                ["train", "params", "optimizer", "params"],
                {"lr": 0.0001, "momentum": 0.9, "nesterov": True},
            ),
            Parameter(["train", "data", "train_fraction"], 0.8),
            Parameter(["train", "data", "training_slides"], None),
            Parameter(["train", "data", "validation_slides"], None),
            Parameter(["train", "model", "architecture"], "mobilenet_v2"),
            Parameter(["train", "data", "patch_size"], 224),
        ],
        grid=[
            ParameterList(
                ["train", "data", "augmentations"],
                [
                    {k: v for d in l for k, v in d.items()}
                    for l in list(
                        product(
                            [
                                {},
                            ],
                            [
                                {
                                    "color_jitter": {
                                        "saturation": 0.6,
                                        "contrast": 0.5,
                                        "brightness": 0.5,
                                        "hue": 0.5,
                                    }
                                },
                                {
                                    "color_jitter": {
                                        "saturation": 0.5,
                                        "contrast": 0.4,
                                        "brightness": 0.4,
                                        "hue": 0.3,
                                    }
                                },
                                {
                                    "color_jitter": {
                                        "saturation": 0.5,
                                        "contrast": 0.3,
                                        "brightness": 0.3,
                                        "hue": 0.1,
                                    }
                                },
                                {},
                            ],
                        )
                    )
                ],
            ),
            ParameterList(
                ["train", "data", "normalizer"],
                [
                    None,
                    {
                        "type": "train",
                        "mean": [0.86489, 0.63272, 0.85928],
                        "std": [0.020820, 0.026320, 0.017309],
                    },
                ],
            ),
        ],
    )
    PretrainingExperiment(name="separate_no_spatial_fine_tune_augmentations").generate(
        fixed=[
            Parameter(["train", "params", "optimizer", "class"], "SGD"),
            Parameter(
                ["train", "params", "optimizer", "params"],
                {"lr": 0.0001, "momentum": 0.9, "nesterov": True},
            ),
            Parameter(["train", "model", "architecture"], "mobilenet_v2"),
        ],
        grid=[
            ParameterList(
                ["train", "data", "augmentations"],
                [
                    {k: v for d in l for k, v in d.items()}
                    for l in list(
                        product(
                            [
                                {
                                    "rotation": {"degrees": 180, "crop": 224},
                                    "flip": None,
                                },
                            ],
                            [
                                {
                                    "color_jitter": {
                                        "saturation": 0.6,
                                        "contrast": 0.5,
                                        "brightness": 0.5,
                                        "hue": 0.5,
                                    }
                                },
                                {
                                    "color_jitter": {
                                        "saturation": 0.5,
                                        "contrast": 0.4,
                                        "brightness": 0.4,
                                        "hue": 0.3,
                                    }
                                },
                                {
                                    "color_jitter": {
                                        "saturation": 0.5,
                                        "contrast": 0.3,
                                        "brightness": 0.3,
                                        "hue": 0.1,
                                    }
                                },
                                {},
                            ],
                        )
                    )
                ],
            ),
            ParameterList(
                ["train", "data", "normalizer"],
                [
                    None,
                    {
                        "type": "train",
                        "mean": [0.86489, 0.63272, 0.85928],
                        "std": [0.020820, 0.026320, 0.017309],
                    },
                ],
            ),
        ],
    )
    PretrainingExperiment(name="separate_spatial_fine_tune_augmentations").generate(
        fixed=[
            Parameter(["train", "params", "optimizer", "class"], "SGD"),
            Parameter(
                ["train", "params", "optimizer", "params"],
                {"lr": 0.0001, "momentum": 0.9, "nesterov": True},
            ),
            Parameter(["train", "model", "architecture"], "mobilenet_v2"),
            Parameter(["train", "data", "patch_size"], 224),
        ],
        grid=[
            ParameterList(
                ["train", "data", "augmentations"],
                [
                    {k: v for d in l for k, v in d.items()}
                    for l in list(
                        product(
                            [
                                {},
                            ],
                            [
                                {
                                    "color_jitter": {
                                        "saturation": 0.6,
                                        "contrast": 0.5,
                                        "brightness": 0.5,
                                        "hue": 0.5,
                                    }
                                },
                                {
                                    "color_jitter": {
                                        "saturation": 0.5,
                                        "contrast": 0.4,
                                        "brightness": 0.4,
                                        "hue": 0.3,
                                    }
                                },
                                {
                                    "color_jitter": {
                                        "saturation": 0.5,
                                        "contrast": 0.3,
                                        "brightness": 0.3,
                                        "hue": 0.1,
                                    }
                                },
                                {},
                            ],
                        )
                    )
                ],
            ),
            ParameterList(
                ["train", "data", "normalizer"],
                [
                    None,
                    {
                        "type": "train",
                        "mean": [0.86489, 0.63272, 0.85928],
                        "std": [0.020820, 0.026320, 0.017309],
                    },
                ],
            ),
        ],
    )
    PretrainingExperiment(name="additional_dropout").generate(
        grid=[
            ParameterList(
                ["train", "model", "dropout"],
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
        ],
    )
    PretrainingExperiment(name="downsampling_factor").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 2000),
        ],
        sequential=[
            [
                ParameterList(
                    ["train", "data", "downsample_factor"], [2.5, 3.5, 4, 4.5, 5]
                )
            ]
        ],
    )
    PretrainingExperiment(name="mobile_freezing").generate(
        fixed=[
            Parameter(
                ["train", "model", "pretrained"],
                True,
            ),
        ],
        sequential=[
            ParameterList(
                ["train", "model", "freeze"], [19, 18, 17, 16, 15, 14, 13, 12, 8, 4, 0]
            )
        ],
    )
    PretrainingExperiment(name="freeze_and_dropout").generate(
        fixed=[
            Parameter(
                ["train", "model", "pretrained"],
                True,
            ),
            Parameter(["train", "model", "freeze"], 17),
            Parameter(["train", "params", "nr_epochs"], 1000),
        ],
        grid=[
            ParameterList(
                ["train", "model", "dropout"],
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
        ],
    )
    PretrainingExperiment(name="freeze_less_and_dropout").generate(
        fixed=[
            Parameter(
                ["train", "model", "pretrained"],
                True,
            ),
            Parameter(["train", "model", "freeze"], 14),
            Parameter(["train", "params", "nr_epochs"], 1000),
        ],
        grid=[
            ParameterList(
                ["train", "model", "dropout"],
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
        ],
    )
    PretrainingExperiment(name="downsample_freeze_dropout").generate(
        fixed=[
            Parameter(
                ["train", "model", "pretrained"],
                True,
            ),
            Parameter(["train", "model", "freeze"], 14),
            Parameter(["train", "params", "nr_epochs"], 1000),
            Parameter(["train", "data", "downsample_factor"], 4.5),
        ],
        grid=[
            ParameterList(
                ["train", "model", "dropout"],
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
        ],
    )
    PretrainingExperiment(name="resnet18_freeze_dropout").generate(
        fixed=[
            Parameter(["train", "model", "architecture"], "resnet18"),
            Parameter(
                ["train", "model", "pretrained"],
                True,
            ),
            Parameter(
                ["train", "model", "dropout"],
                0.5,
            ),
        ],
        sequential=[ParameterList(["train", "model", "freeze"], [0, 5, 6, 7, 8])],
    )
    PretrainingExperiment(name="resnet34_freeze_dropout").generate(
        fixed=[
            Parameter(["train", "model", "architecture"], "resnet34"),
            Parameter(
                ["train", "model", "pretrained"],
                True,
            ),
            Parameter(
                ["train", "model", "dropout"],
                0.5,
            ),
        ],
        sequential=[ParameterList(["train", "model", "freeze"], [0, 5, 6, 7, 8])],
    )
    PretrainingExperiment(name="baseline", queue="prod.long").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 2000),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "batch_size"],
                [32],
            ),
        ],
    )
    PretrainingExperiment(name="optimizer_long", queue="prod.long").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 2000),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "optimizer"],
                [
                    {"class": "Adam", "params": {"lr": 0.0001}},
                    {"class": "AdamW", "params": {"lr": 0.0001}},
                    {
                        "class": "SGD",
                        "params": {"lr": 0.0001, "momentum": 0.9, "nesterov": True},
                    },
                ],
            )
        ],
    )
    PretrainingExperiment(name="batch_sizes_long", queue="prod.long").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 2000),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "batch_size"],
                [8, 16, 64, 128],
            ),
        ],
    )
    PretrainingExperiment(name="step_lr").generate(
        fixed=[
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-4),
        ],
        sequential=[
            [
                ParameterList(
                    ["train", "params", "optimizer", "scheduler"],
                    [None]
                    + [
                        {
                            "class": "StepLR",
                            "params": {"step_size": step_size, "gamma": gamma},
                        }
                        for step_size in [50, 75, 100]
                        for gamma in [0.5, 0.2, 0.1]
                    ],
                )
            ]
        ],
    )
    PretrainingExperiment(name="exponential_lr").generate(
        fixed=[
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-4),
        ],
        sequential=[
            [
                ParameterList(
                    ["train", "params", "optimizer", "scheduler"],
                    [None]
                    + [
                        {"class": "ExponentialLR", "params": {"gamma": gamma}}
                        for gamma in [0.999, 0.99, 0.95]
                    ],
                )
            ]
        ],
    )

    CNNTestingExperiment(name="exponential_lr_decay").generate(
        fixed=[
            Parameter(
                ["test", "model", "architecture"],
                "s3://mlflow/633/02906fe539444b13a76d39d4a0dfbb6f/artifacts/best.valid.MultiLabelBalancedAccuracy",
            )
        ],
        sequential=[ParameterList(["test", "params", "overlap"], [150, 175, 200, 210])],
    )
    CNNTestingExperiment(name="step_lr_decay").generate(
        fixed=[
            Parameter(
                ["test", "model", "architecture"],
                "s3://mlflow/633/c62233eed1574d2ca2d9b8ee51b83ffc/artifacts/best.valid.MultiLabelBalancedAccuracy",
            )
        ],
        sequential=[ParameterList(["test", "params", "overlap"], [150, 175, 200, 210])],
    )
    CNNTestingExperiment(name="batch_size_and_sgd").generate(
        fixed=[
            Parameter(
                ["test", "model", "architecture"],
                "s3://mlflow/633/19a9b40d174f40c4b217ddf84eb63e3b/artifacts/best.valid.MultiLabelBalancedAccuracy",
            )
        ],
        sequential=[ParameterList(["test", "params", "overlap"], [150, 175, 200, 210])],
    )
    CNNTestingExperiment(name="various").generate(
        sequential=[
            ParameterList(
                ["test", "model", "architecture"],
                [
                    f"s3://mlflow/633/{run}/artifacts/best.valid.MultiLabelBalancedAccuracy"
                    for run in [
                        "19a9b40d174f40c4b217ddf84eb63e3b",
                        "c62233eed1574d2ca2d9b8ee51b83ffc",
                        "02906fe539444b13a76d39d4a0dfbb6f",
                    ]
                ],
            )
        ]
    )
    CNNTestingExperiment(name="thresholds").generate(
        fixed=[
            Parameter(
                ["test", "model", "architecture"],
                "s3://mlflow/633/c62233eed1574d2ca2d9b8ee51b83ffc/artifacts/best.valid.MultiLabelBalancedAccuracy",
            )
        ],
        sequential=[
            ParameterList(
                ["test", "params", "threshold"],
                [0.0, 0.1, 0.15, 0.2, 0.225, 0.25, 0.275, 0.3, 0.323, 0.35],
            )
        ],
    )

    GNNTestingExperiment(name="gnn_deep").generate(
        sequential=[
            ParameterList(
                ["test", "model", "architecture"],
                [
                    f"s3://mlflow/631/{run}/artifacts/best.valid.segmentation.MeanIoU"
                    for run in [
                        "08d570cf39eb435281679a3754c21dce",
                        "3cb5cf60d3e0479a929c4a6ce9aee24b",
                        "b8df33ef727c43bfb3212b1905d1ed48",
                        "38a6507ca9014b99bf0b43978ea248cf",
                    ]
                ],
            )
        ]
    )
    GNNTestingExperiment(name="deep_baseline").generate(
        sequential=[
            ParameterList(
                ["test", "model", "architecture"],
                [
                    f"s3://mlflow/631/{run}/artifacts/best.valid.segmentation.MeanIoU"
                    for run in [
                        "029a0ad7cb2043afbb66be6af32a7ddc",
                        "eab6a4e236e74563938ea992a685289c",
                        "0da677d6412f449db8d1e5bade323cbc",
                        "fa451fd2a6014fed87c53fff40d2beab",
                    ]
                ],
            )
        ]
    )
    GNNTestingExperiment(name="thresholds").generate(
        fixed=[
            Parameter(
                ["test", "model", "architecture"],
                "s3://mlflow/631/3cb5cf60d3e0479a929c4a6ce9aee24b/artifacts/best.valid.segmentation.MeanIoU",
            )
        ],
        sequential=[
            ParameterList(
                ["test", "params", "threshold"],
                [0.0, 0.1, 0.15, 0.2, 0.225, 0.25, 0.275, 0.3, 0.323, 0.35],
            )
        ],
    )
