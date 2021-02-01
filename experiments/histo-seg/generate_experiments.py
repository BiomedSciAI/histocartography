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
                Parameter(["feature_extraction", "params", "num_workers"], 0),
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
            main_file="test_gnn",
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
        name="augmented_new_pretrained",
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
            Parameter(
                ["feature_extraction", "params", "architecture"],
                "models/19a9b40d174f40c4b217ddf84eb63e3b_best_valid_MultiLabelBalancedAccuracy.pth",
            ),
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
                    [
                        f"v4_19a9b40d174f40c4b217ddf84eb63e3b_{s}"
                        for s in [300, 600, 900, 1200]
                    ],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
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
    CPUPreprocessingExperiment(
        name="two_hop_augmented_new_pretrained",
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
            Parameter(["graph_builders", "params", "hops"], 2),
            Parameter(
                ["feature_extraction", "params", "architecture"],
                "models/19a9b40d174f40c4b217ddf84eb63e3b_best_valid_MultiLabelBalancedAccuracy.pth",
            ),
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
                    [
                        f"v5_two_hop_19a9b40d174f40c4b217ddf84eb63e3b_{s}"
                        for s in [300, 600, 900, 1200]
                    ],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="two_hop_augmented_new_baseline",
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
            Parameter(["graph_builders", "params", "hops"], 2),
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
                    [f"v5_two_hop_mobilenet_{s}" for s in [300, 600, 900, 1200]],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="three_hop_augmented_new_pretrained",
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
            Parameter(["graph_builders", "params", "hops"], 3),
            Parameter(
                ["feature_extraction", "params", "architecture"],
                "models/19a9b40d174f40c4b217ddf84eb63e3b_best_valid_MultiLabelBalancedAccuracy.pth",
            ),
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
                    [
                        f"v5_three_hop_19a9b40d174f40c4b217ddf84eb63e3b_{s}"
                        for s in [300, 600, 900, 1200]
                    ],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="three_hop_augmented_new_baseline",
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
            Parameter(["graph_builders", "params", "hops"], 3),
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
                    [f"v5_three_hop_mobilenet_{s}" for s in [300, 600, 900, 1200]],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="grid_graph", base="config/augmented_preprocess.yml"
    ).generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            )
        ],
        sequential=[
            [
                ParameterList(
                    ["superpixel", "params", "nr_superpixels"], [300, 800, 60, 200]
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"], [1000, 1000, 1000, 1000]
                ),
                ParameterList(
                    ["feature_extraction", "params", "size"], [224, 224, 448, 448]
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [
                        f"v6_grid_mobilenet_{s}"
                        for s in [
                            "40x_no_overlap",
                            "40x_overlap",
                            "20x_no_overlap",
                            "40x_overlap",
                        ]
                    ],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="non_overlapping_graph", base="config/augmented_preprocess.yml"
    ).generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            )
        ],
        sequential=[
            [
                ParameterList(
                    ["superpixel", "params", "nr_superpixels"], [700, 400, 250]
                ),
                ParameterList(["superpixel", "params", "compactness"], [30, 30, 30]),
                ParameterList(
                    ["feature_extraction", "params", "size"], [224, 336, 448]
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [
                        f"v6_mobilenet_{s}"
                        for s in ["40x_no_overlap", "30x_no_overlap", "20x_no_overlap"]
                    ],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        "color_merged_low_no_overlap", base="config/merged_preprocess.yml"
    ).generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
            Parameter(["superpixel", "params", "threshold"], 0.01),
        ],
        sequential=[
            [
                ParameterList(
                    ["superpixel", "params", "nr_superpixels"],
                    [700, 400, 250],
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"],
                    [30, 30, 30],
                ),
                ParameterList(
                    ["feature_extraction", "params", "downsample_factor"],
                    [1, 2, 3],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [
                        f"v10_mobilenet_low_{s}"
                        for s in ["40x_no_overlap", "30x_no_overlap", "20x_no_overlap"]
                    ],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        "color_merged_med_no_overlap", base="config/merged_preprocess.yml"
    ).generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
            Parameter(["superpixel", "params", "threshold"], 0.02),
        ],
        sequential=[
            [
                ParameterList(
                    ["superpixel", "params", "nr_superpixels"],
                    [700, 400, 250],
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"],
                    [30, 30, 30],
                ),
                ParameterList(
                    ["feature_extraction", "params", "downsample_factor"],
                    [1, 2, 3],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [
                        f"v10_mobilenet_med_{s}"
                        for s in ["40x_no_overlap", "30x_no_overlap", "20x_no_overlap"]
                    ],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        "color_merged_high_no_overlap", base="config/merged_preprocess.yml"
    ).generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
            Parameter(["superpixel", "params", "threshold"], 0.03),
        ],
        sequential=[
            [
                ParameterList(
                    ["superpixel", "params", "nr_superpixels"],
                    [700, 400, 250],
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"],
                    [30, 30, 30],
                ),
                ParameterList(
                    ["feature_extraction", "params", "downsample_factor"],
                    [1, 2, 3],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [
                        f"v10_mobilenet_high_{s}"
                        for s in ["40x_no_overlap", "30x_no_overlap", "20x_no_overlap"]
                    ],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        "color_merged_very_high_no_overlap", base="config/merged_preprocess.yml"
    ).generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
            Parameter(["superpixel", "params", "threshold"], 0.04),
        ],
        sequential=[
            [
                ParameterList(
                    ["superpixel", "params", "nr_superpixels"],
                    [700, 400, 250],
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"],
                    [30, 30, 30],
                ),
                ParameterList(
                    ["feature_extraction", "params", "downsample_factor"],
                    [1, 2, 3],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [
                        f"v10_mobilenet_very_high_{s}"
                        for s in ["40x_no_overlap", "30x_no_overlap", "20x_no_overlap"]
                    ],
                ),
            ]
        ],
    )
    GPUPreprocessingExperiment(name="v10_no", base="config/augmented_preprocess.yml").generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
        ],
        sequential=[
            [
                ParameterList(
                    ["superpixel", "params", "nr_superpixels"],
                    [700, 400, 250],
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"],
                    [30, 30, 30],
                ),
                ParameterList(
                    ["feature_extraction", "params", "downsample_factor"],
                    [1, 2, 3],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [
                        f"v10_mobilenet_no_{s}"
                        for s in ["40x_no_overlap", "30x_no_overlap", "20x_no_overlap"]
                    ],
                ),
            ]
        ],
    )
    GPUPreprocessingExperiment(name="v11_standard", base="config/new_feature.yml").generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
            Parameter(
                ["feature_extraction", "params", "patch_size"],
                224,
            ),
            Parameter(
                ["feature_extraction", "params", "stride"],
                32,
            ),
        ],
        sequential=[
            ParameterList(
                ["feature_extraction", "params", "downsample_factor"],
                [1, 2, 3, 4],
            ),
        ]
    )
    GPUPreprocessingExperiment(name="v11_less_context", base="config/new_feature.yml").generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
            Parameter(
                ["feature_extraction", "params", "patch_size"],
                112,
            ),
            Parameter(
                ["feature_extraction", "params", "stride"],
                32,
            ),
        ],
        sequential=[
            ParameterList(
                ["feature_extraction", "params", "downsample_factor"],
                [1, 2, 3, 4],
            ),
        ]
    )
    GPUPreprocessingExperiment(name="v11_more_finegrained", base="config/new_feature.yml", queue="prod.long").generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
            Parameter(
                ["feature_extraction", "params", "patch_size"],
                224,
            ),
            Parameter(
                ["feature_extraction", "params", "stride"],
                16,
            ),
        ],
        sequential=[
            ParameterList(
                ["feature_extraction", "params", "downsample_factor"],
                [2, 3, 4],
            ),
        ]
    )
    CPUPreprocessingExperiment(name="v11_standard_low", base="config/new_preprocess.yml").generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
            Parameter(
                ["feature_extraction", "params", "patch_size"],
                224,
            ),
            Parameter(
                ["feature_extraction", "params", "stride"],
                32,
            ),
            Parameter(["superpixel", "params", "threshold"], 0.01),
        ],
        sequential=[
            [
                ParameterList(
                    ["superpixel", "params", "nr_superpixels"],
                    [700, 400, 250],
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"],
                    [30, 30, 30],
                ),
                ParameterList(
                    ["feature_extraction", "params", "downsample_factor"],
                    [2, 3, 4],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [
                        f"v11_mobilenet_low_{s}"
                        for s in ["20x", "13x", "10x"]
                    ],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(name="v11_standard_med", base="config/new_preprocess.yml").generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
            Parameter(
                ["feature_extraction", "params", "patch_size"],
                224,
            ),
            Parameter(
                ["feature_extraction", "params", "stride"],
                32,
            ),
            Parameter(["superpixel", "params", "threshold"], 0.02),
        ],
        sequential=[
            [
                ParameterList(
                    ["superpixel", "params", "nr_superpixels"],
                    [700, 400, 250],
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"],
                    [30, 30, 30],
                ),
                ParameterList(
                    ["feature_extraction", "params", "downsample_factor"],
                    [2, 3, 4],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [
                        f"v11_mobilenet_med_{s}"
                        for s in ["20x", "13x", "10x"]
                    ],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(name="v11_standard_high", base="config/new_preprocess.yml").generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
            Parameter(
                ["feature_extraction", "params", "patch_size"],
                224,
            ),
            Parameter(
                ["feature_extraction", "params", "stride"],
                32,
            ),
            Parameter(["superpixel", "params", "threshold"], 0.03),
        ],
        sequential=[
            [
                ParameterList(
                    ["superpixel", "params", "nr_superpixels"],
                    [700, 400, 250],
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"],
                    [30, 30, 30],
                ),
                ParameterList(
                    ["feature_extraction", "params", "downsample_factor"],
                    [2, 3, 4],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [
                        f"v11_mobilenet_high_{s}"
                        for s in ["20x", "13x", "10x"]
                    ],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(name="v11_standard_very_high", base="config/new_preprocess.yml").generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
            Parameter(
                ["feature_extraction", "params", "patch_size"],
                224,
            ),
            Parameter(
                ["feature_extraction", "params", "stride"],
                32,
            ),
            Parameter(["superpixel", "params", "threshold"], 0.04),
        ],
        sequential=[
            [
                ParameterList(
                    ["superpixel", "params", "nr_superpixels"],
                    [700, 400, 250],
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"],
                    [30, 30, 30],
                ),
                ParameterList(
                    ["feature_extraction", "params", "downsample_factor"],
                    [2, 3, 4],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [
                        f"v11_mobilenet_very_high_{s}"
                        for s in ["20x", "13x", "10x"]
                    ],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(name="v11_standard_no", base="config/new_preprocess_no_merge.yml").generate(
        fixed=[
            Parameter(
                ["stain_normalizers", "params", "target"],
                "ZT111_4_C_7_1",
            ),
            Parameter(
                ["feature_extraction", "params", "patch_size"],
                224,
            ),
            Parameter(
                ["feature_extraction", "params", "stride"],
                32,
            ),
        ],
        sequential=[
            [
                ParameterList(
                    ["superpixel", "params", "nr_superpixels"],
                    [700, 400, 250],
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"],
                    [30, 30, 30],
                ),
                ParameterList(
                    ["feature_extraction", "params", "downsample_factor"],
                    [2, 3, 4],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [
                        f"v11_mobilenet_no_{s}"
                        for s in ["20x", "13x", "10x"]
                    ],
                ),
            ]
        ],
    )

    # ETH
    GraphClassifierExperiment(name="multihop").generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
            Parameter(["train", "model", "graph_classifier_config"], None),
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
            Parameter(["train", "data", "augmentation_mode"], "graph"),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                12,
            ),
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-4),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {"class": "ExponentialLR", "params": {"gamma": 0.99}},
            ),
        ],
        sequential=[
            [
                ParameterList(
                    ["train", "data", "graph_directory"],
                    [
                        f"outputs/{x}"
                        for x in [
                            "v5_three_hop_mobilenet_300",
                            "v5_two_hop_mobilenet_300",
                            "v4_mobilenet_300",
                            "v5_three_hop_mobilenet_600",
                            "v5_two_hop_mobilenet_600",
                            "v4_mobilenet_600",
                        ]
                    ],
                )
            ]
        ],
    )
    GraphClassifierExperiment(name="v9_color_merged").generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
            Parameter(["train", "model", "graph_classifier_config"], None),
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
            Parameter(["train", "data", "augmentation_mode"], "graph"),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                12,
            ),
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-4),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {"class": "ExponentialLR", "params": {"gamma": 0.99}},
            ),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v9_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "40x_no_overlap",
                        "30x_no_overlap",
                        "20x_no_overlap",
                    ]
                    for level in ["low", "med", "high", "very_high"]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v8_edge_merged").generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
            Parameter(["train", "model", "graph_classifier_config"], None),
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
            Parameter(["train", "data", "augmentation_mode"], "graph"),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                12,
            ),
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-4),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {"class": "ExponentialLR", "params": {"gamma": 0.99}},
            ),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v8_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "40x_no_overlap",
                        "30x_no_overlap",
                        "20x_no_overlap",
                    ]
                    for level in ["low", "med", "high"]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v10_old_image_level").generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
            Parameter(["train", "model", "node_classifier_config"], None),
            Parameter(
                ["train", "model", "gnn_config", "dropout"],
                0.5,
            ),
            Parameter(
                ["train", "model", "gnn_config", "hidden_dim"],
                32,
            ),
            Parameter(
                ["train", "model", "gnn_config", "output_dim"],
                32,
            ),
            Parameter(["train", "data", "augmentation_mode"], "graph"),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                12,
            ),
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-4),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {"class": "ExponentialLR", "params": {"gamma": 0.99}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "original_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level")
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v10_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "40x_no_overlap",
                        "30x_no_overlap",
                        "20x_no_overlap",
                    ]
                    for level in ["low", "med", "high", "very_high"]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v10_new_image_level").generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
            Parameter(["train", "model", "node_classifier_config"], None),
            Parameter(
                ["train", "model", "gnn_config", "dropout"],
                0.5,
            ),
            Parameter(
                ["train", "model", "gnn_config", "hidden_dim"],
                32,
            ),
            Parameter(
                ["train", "model", "gnn_config", "output_dim"],
                32,
            ),
            Parameter(["train", "data", "augmentation_mode"], "graph"),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                12,
            ),
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-4),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {"class": "ExponentialLR", "params": {"gamma": 0.99}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "new_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level")
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v10_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "40x_no_overlap",
                        "30x_no_overlap",
                        "20x_no_overlap",
                    ]
                    for level in ["low", "med", "high", "very_high"]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v10_image_level_dropout").generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
            Parameter(["train", "model", "node_classifier_config"], None),
            Parameter(
                ["train", "model", "gnn_config", "dropout"],
                0.5,
            ),
            Parameter(
                ["train", "model", "gnn_config", "hidden_dim"],
                32,
            ),
            Parameter(
                ["train", "model", "gnn_config", "output_dim"],
                32,
            ),
            Parameter(["train", "data", "augmentation_mode"], "graph"),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                12,
            ),
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-4),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {"class": "ExponentialLR", "params": {"gamma": 0.99}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "new_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level")
        ],
        sequential=[
            ParameterList(["train", "model", "graph_classifier_config", "input_dropout"], [0.0, 0.3, 0.5, 0.7])
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v10_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "20x_no_overlap",
                    ]
                    for level in ["low", "med", "high", "very_high"]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v10_gnn_layers").generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
            Parameter(["train", "model", "node_classifier_config"], None),
            Parameter(
                ["train", "model", "gnn_config", "dropout"],
                0.5,
            ),
            Parameter(
                ["train", "model", "gnn_config", "hidden_dim"],
                32,
            ),
            Parameter(
                ["train", "model", "gnn_config", "output_dim"],
                32,
            ),
            Parameter(["train", "model", "graph_classifier_config", "input_dropout"], 0.3),
            Parameter(["train", "data", "augmentation_mode"], "graph"),
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-4),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {"class": "ExponentialLR", "params": {"gamma": 0.99}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "new_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level")
        ],
        sequential=[
            ParameterList(
                ["train", "model", "gnn_config", "n_layers"],
                [2, 3, 4, 6, 8, 12, 16],
            ),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v10_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "40x_no_overlap",
                        "30x_no_overlap",
                        "20x_no_overlap",
                    ]
                    for level in ["high"]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v10_augmentation").generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
            Parameter(["train", "model", "node_classifier_config"], None),
            Parameter(
                ["train", "model", "gnn_config", "dropout"],
                0.5,
            ),
            Parameter(
                ["train", "model", "gnn_config", "hidden_dim"],
                32,
            ),
            Parameter(
                ["train", "model", "gnn_config", "output_dim"],
                32,
            ),
            Parameter(["train", "model", "graph_classifier_config", "input_dropout"], 0.3),
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-4),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {"class": "ExponentialLR", "params": {"gamma": 0.99}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "new_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                6,
            ),
        ],
        sequential=[
            ParameterList(["train", "data", "augmentation_mode"], [None, "graph", "node"]),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v10_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "40x_no_overlap",
                        "30x_no_overlap",
                        "20x_no_overlap",
                    ]
                    for level in ["high"]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v10_lr").generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
            Parameter(["train", "model", "node_classifier_config"], None),
            Parameter(
                ["train", "model", "gnn_config", "dropout"],
                0.5,
            ),
            Parameter(
                ["train", "model", "gnn_config", "hidden_dim"],
                32,
            ),
            Parameter(
                ["train", "model", "gnn_config", "output_dim"],
                32,
            ),
            Parameter(["train", "model", "graph_classifier_config", "input_dropout"], 0.3),
            Parameter(["train", "data", "image_labels_mode"], "new_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                6,
            ),
            Parameter(["train", "data", "augmentation_mode"], "graph"),
        ],
        sequential=[
            ParameterList(["train", "params", "optimizer", "params", "lr"], [1e-3, 1e-4, 1e-5]),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v10_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "40x_no_overlap",
                        "30x_no_overlap",
                        "20x_no_overlap",
                    ]
                    for level in ["high"]
                ],
            )
        ],
    )
    GraphClassifierExperiment(name="v11_old_image_level").generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
            Parameter(["train", "model", "node_classifier_config"], None),
            Parameter(
                ["train", "model", "gnn_config", "dropout"],
                0.5,
            ),
            Parameter(
                ["train", "model", "gnn_config", "hidden_dim"],
                32,
            ),
            Parameter(
                ["train", "model", "gnn_config", "output_dim"],
                32,
            ),
            Parameter(["train", "data", "augmentation_mode"], "graph"),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                12,
            ),
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-4),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {"class": "ExponentialLR", "params": {"gamma": 0.99}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "original_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level")
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v11_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "20x",
                        "13x",
                        "10x",
                    ]
                    for level in ["low", "med", "high", "very_high"]
                ],
            )
        ],
    )

    # Pretraining
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
    PretrainingExperiment(name="drop_patches").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-3),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "MultiStepLR",
                    "params": {"milestones": [50, 100, 200, 300], "gamma": 0.2},
                },
            ),
        ],
        grid=[
            ParameterList(["train", "data", "drop_multiclass_patches"], [True, False]),
            ParameterList(
                ["train", "data", "drop_tissue_patches"],
                [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
            ),
        ],
    )
    PretrainingExperiment(name="drop_patches_on_val").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-3),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "MultiStepLR",
                    "params": {"milestones": [50, 100, 200, 300], "gamma": 0.2},
                },
            ),
            Parameter(["train", "data", "drop_validation_patches"], True),
        ],
        grid=[
            ParameterList(["train", "data", "drop_multiclass_patches"], [True, False]),
            ParameterList(
                ["train", "data", "drop_tissue_patches"],
                [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
            ),
        ],
    )
    PretrainingExperiment(name="drop_unlabelled").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-3),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "MultiStepLR",
                    "params": {"milestones": [50, 100, 200, 300], "gamma": 0.2},
                },
            ),
            Parameter(["train", "data", "drop_unlabelled_patches"], True),
        ],
        grid=[
            ParameterList(
                ["train", "data", "drop_tissue_patches"],
                [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
            )
        ],
    )
    PretrainingExperiment(name="balanced_batches").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-3),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "MultiStepLR",
                    "params": {"milestones": [50, 100, 200, 300], "gamma": 0.2},
                },
            ),
            Parameter(["train", "params", "balanced_batches"], True),
        ],
        grid=[
            ParameterList(
                ["train", "data", "drop_tissue_patches"], [0.0, 0.15, 0.3, 0.45]
            ),
            ParameterList(["train", "data", "drop_unlabelled_patches"], [True, False]),
            ParameterList(["train", "data", "drop_multiclass_patches"], [True, False]),
        ],
    )
    PretrainingExperiment(name="encoder_pretraining").generate(
        fixed=[
            Parameter(["train", "params", "nr_epochs"], 500),
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-3),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "MultiStepLR",
                    "params": {"milestones": [50, 100, 200, 300], "gamma": 0.2},
                },
            ),
            Parameter(["train", "params", "balanced_batches"], True),
            Parameter(["train", "data", "drop_tissue_patches"], 0.15),
            Parameter(["train", "data", "drop_unlabelled_patches"], True),
            Parameter(["train", "data", "drop_multiclass_patches"], False),
        ],
        grid=[
            ParameterList(
                ["train", "params", "pretrain_epochs"], [None, 3, 5, 10, 17, 25]
            ),
            ParameterList(["train", "model", "freeze"], [17, 14, 12, 0]),
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
    CNNTestingExperiment(name="normal_loader").generate(
        sequential=[
            ParameterList(
                ["test", "model", "architecture"],
                [
                    f"s3://mlflow/633/{run}/artifacts/best.valid.MultiLabelBalancedAccuracy"
                    for run in [
                        "69c7f3f2bf3b4bb6844c8b039496cf84",
                    ]
                ],
            )
        ],
        grid=[
            ParameterList(
                ["test", "params", "inference_mode"], ["patch_based", "hacky"]
            )
        ],
    )
    CNNTestingExperiment(name="balanced_loader").generate(
        sequential=[
            ParameterList(
                ["test", "model", "architecture"],
                [
                    f"s3://mlflow/633/{run}/artifacts/best.valid.MultiLabelBalancedAccuracy"
                    for run in [
                        "4097a5e9d82244148f0036ce45cf913a",
                    ]
                ],
            )
        ],
        grid=[
            ParameterList(
                ["test", "params", "inference_mode"], ["patch_based", "hacky"]
            )
        ],
    )
    CNNTestingExperiment(name="classifier_pretrain").generate(
        sequential=[
            ParameterList(
                ["test", "model", "architecture"],
                [
                    f"s3://mlflow/633/{run}/artifacts/best.valid.MultiLabelBalancedAccuracy"
                    for run in [
                        "51e964834a214c0cb5761abd233d5277",
                    ]
                ],
            )
        ],
        grid=[
            ParameterList(
                ["test", "params", "inference_mode"], ["patch_based", "hacky"]
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
    GNNTestingExperiment(name="deep_gnns").generate(
        sequential=[
            ParameterList(
                ["test", "model", "architecture"],
                [
                    f"s3://mlflow/631/{run}/artifacts/best.valid.segmentation.MeanIoU"
                    for run in [
                        "05466db6787b474099b7c3c675929e80",
                        "bbc33034429b423da734ff0e5d5b8c15",
                        "1ca94f631e8d4cb9b1f45e4030f57c89",
                    ]
                ],
            )
        ]
    )
    GNNTestingExperiment(name="magnification_no_overlap").generate(
        sequential=[
            ParameterList(
                ["test", "model", "architecture"],
                [
                    f"s3://mlflow/631/{run}/artifacts/best.valid.segmentation.MeanIoU"
                    for run in [
                        "8fa60ec176b94c4a8c1bbbf2f3b4b393",
                        "7d6a6336892a4741bf2ef63331b36656",
                        "8a29f56a30924e43bedae3a998a7cbfd",
                    ]
                ],
            )
        ]
    )
    GNNTestingExperiment(name="grid_graph").generate(
        sequential=[
            ParameterList(
                ["test", "model", "architecture"],
                [
                    f"s3://mlflow/631/{run}/artifacts/best.valid.segmentation.MeanIoU"
                    for run in [
                        "f8c2c2772e6c4f7daf717ab792c87560",
                        "2e0990b4560442b78af804816945d5c2",
                    ]
                ],
            )
        ]
    )
    GNNTestingExperiment(name="exponential_lr").generate(
        sequential=[
            ParameterList(
                ["test", "model", "architecture"],
                [
                    f"s3://mlflow/631/{run}/artifacts/best.valid.segmentation.MeanIoU"
                    for run in [
                        "6ac7eccc11c848738a747270db669584",
                    ]
                ],
            )
        ]
    )
    GNNTestingExperiment(name="one_vs_two_hop_300").generate(
        sequential=[
            ParameterList(
                ["test", "model", "architecture"],
                [
                    f"s3://mlflow/631/{run}/artifacts/best.valid.segmentation.MeanIoU"
                    for run in [
                        "d6cf9a9f69d9414c8528bc142b19d364",
                        "f10989d99f4448ae8b1eb93bff8e2032",
                    ]
                ],
            )
        ]
    )
    GNNTestingExperiment(name="color_merged").generate(
        sequential=[
            ParameterList(
                ["test", "model", "architecture"],
                [
                    f"s3://mlflow/631/{run}/artifacts/best.valid.segmentation.MeanIoU"
                    for run in [
                        "4128202202f04ccaa7727b6d6627feaa",
                        "a9f0461d451d4f5bb28edd0a9ec672a2",
                        "15c6b2681a504572b89b30b16ac9ea38",
                        "32e6c6a331534f3f9fa251a47b88fc8d",
                        "a27b3e2078f04ef39f8add1028d4575c",
                        "4fefe6e1b4104f038edd509a7f1cecb5",
                        "19211a3e4e2043e888cc6ad65ddabb77",
                        "a93c55d7a825431e8c03db8849401cfb",
                        "910e4bc1b7d84c299b676eed8e443f3c",
                        "a48e8fb2496f47c78a7b7b2edbb12dd5",
                        "615b85e22bff492387226a16326604f0",
                    ]
                ],
            )
        ]
    )
