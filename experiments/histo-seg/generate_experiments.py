import argparse
import copy
import logging
import shutil
from dataclasses import dataclass
from functools import reduce
from itertools import product
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Union

import numpy as np
import yaml

BASE = "default.yml"
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
    def _update_config(config, path, value):
        if len(path) > 0:
            if not Experiment._path_exists(config, path):
                logging.warning(
                    f"Config path {path} does not exist. This might be an error"
                )
            reduce(dict.get, path[:-1], config).update({path[-1]: value})

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
    def __init__(self, name, queue="prod.med") -> None:
        super().__init__(
            "pretraining_" + name,
            cores=2,
            core_multiplier=6,
            gpus=1,
            subsample=None,
            main_file="pretrain",
            queue=queue,
            disable_multithreading=False,
            no_save=False,
            base="pretrain.yml",
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
                    16,
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
            base="mnist.yml",
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
    def __init__(self, name, cores=4, queue="prod.med") -> None:
        super().__init__(
            "preprocessing_" + name,
            cores=cores,
            core_multiplier=7,
            gpus=0,
            subsample=None,
            main_file="preprocess",
            queue=queue,
            disable_multithreading=True,
            no_save=False,
            base="default.yml",
        )
        self.name = name
        self.cores = cores

    def generate(self, fixed: Iterable[ParameterList] = list(), **kwargs):
        super().generate(
            [
                Parameter(["preprocess", "params", "cores"], self.cores * 7),
            ]
            + fixed,
            **kwargs,
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
            base="default.yml",
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


def generate_performance_test(path: str, base: str):
    with open(base) as file:
        config: dict = yaml.load(file, Loader=yaml.FullLoader)

    job_name = "scaling_test"
    job_id = 0
    subsample = 256
    for cores in [1, 2, 4]:
        for threads_per_core in [1, 2, 4, 8]:
            # Generate config
            new_config = config.copy()
            new_config["preprocess"]["params"]["cores"] = cores * threads_per_core
            new_config["preprocess"]["stages"]["superpixel_extractor"]["params"] = {
                "nr_superpixels": 100,
                "color_space": "rgb",
                "downsampling_factor": 4,
            }

            # Generate lsf file
            lsf_content = get_lsf(
                config_name=f"job{job_id}",
                queue="prod.short",
                cores=cores,
                log_name=f"{job_name}{job_id}",
                nosave=True,
                subsample=subsample,
                disable_multithreading=True,
            )

            # Write files
            target_directory = Path(path) / job_name
            if not target_directory.exists():
                target_directory.mkdir()
            with open(target_directory / f"job{job_id}.lsf", "w") as file:
                file.write(lsf_content)
            with open(target_directory / f"job{job_id}.yml", "w") as file:
                yaml.dump(new_config, file)

            job_id += 1


def generate_upper_bounds(path: str, base: str):
    with open(base) as file:
        config: dict = yaml.load(file, Loader=yaml.FullLoader)

    job_name = "upper_bound_test"
    job_id = 0
    cores = 6
    for nr_superpixels in [100, 250, 500, 1000, 2000, 4000, 8000]:
        # Generate config
        new_config = config.copy()
        new_config["upper_bound"]["params"]["cores"] = cores * 6
        new_config["upper_bound"]["stages"]["superpixel_extractor"]["params"] = {
            "nr_superpixels": nr_superpixels,
        }

        # Generate lsf file
        lsf_content = get_lsf(
            config_name=f"job{job_id}",
            queue="prod.med",
            cores=cores,
            log_name=f"{job_name}{job_id}",
            main_file_name="upper_bound",
            disable_multithreading=True,
        )

        # Write files
        target_directory = Path(path) / job_name
        if not target_directory.exists():
            target_directory.mkdir()
        with open(target_directory / f"job{job_id}.lsf", "w") as file:
            file.write(lsf_content)
        with open(target_directory / f"job{job_id}.yml", "w") as file:
            yaml.dump(new_config, file)

        job_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="/Users/anv/Documents/experiment_configs"
    )
    parser.add_argument("--base", type=str, default="default.yml")
    args = parser.parse_args()

    PATH = args.path
    BASE = args.base

    generate_performance_test(path=args.path, base=args.base)
    generate_upper_bounds(path=args.path, base=args.base)

    # Preprocessing
    PreprocessingExperiment(name="superpixels").generate(
        sequential=[
            ParameterList(
                [
                    "preprocess",
                    "stages",
                    "superpixel_extractor",
                    "params",
                    "nr_superpixels",
                ],
                [100, 250, 500, 1000, 2000, 4000, 8000],
            )
        ],
    )
    PreprocessingExperiment(name="handcrafted",).generate(
        fixed=[
            Parameter(
                ["preprocess", "stages", "feature_extractor"],
                {"class": "HandcraftedFeatureExtractor"},
            ),
        ],
        sequential=[
            ParameterList(
                [
                    "preprocess",
                    "stages",
                    "superpixel_extractor",
                    "params",
                    "nr_superpixels",
                ],
                [500, 1000, 2000],
            )
        ],
    )
    PreprocessingExperiment(name="resnet34",).generate(
        fixed=[
            Parameter(
                ["preprocess", "stages", "feature_extractor", "params", "architecture"],
                "resnet34",
            ),
        ],
        sequential=[
            ParameterList(
                [
                    "preprocess",
                    "stages",
                    "superpixel_extractor",
                    "params",
                    "nr_superpixels",
                ],
                [500, 1000, 2000],
            )
        ],
    )
    PreprocessingExperiment(name="lowres",).generate(
        fixed=[
            Parameter(
                ["preprocess", "stages", "feature_extractor", "params", "architecture"],
                "resnet34",
            ),
        ],
        grid=[
            ParameterList(
                [
                    "preprocess",
                    "stages",
                    "superpixel_extractor",
                    "params",
                    "nr_superpixels",
                ],
                [500, 1000],
            ),
            ParameterList(
                ["preprocess", "stages", "feature_extractor", "params", "size"],
                [336, 448, 672],
            ),
        ],
    )
    PreprocessingExperiment(name="v1_few_superpixels",).generate(
        fixed=[
            Parameter(["preprocess", "params", "only_superpixel"], False),
        ],
        grid=[
            ParameterList(
                [
                    "preprocess",
                    "stages",
                    "superpixel_extractor",
                    "params",
                    "nr_superpixels",
                ],
                [100, 200, 300, 400, 600, 800],
            ),
            ParameterList(
                ["preprocess", "stages", "feature_extractor", "params", "architecture"],
                ["resnet18", "resnet34", "resnet50"],
            ),
        ],
    )
    PreprocessingExperiment(name="v1",).generate(
        grid=[
            ParameterList(
                [
                    "preprocess",
                    "stages",
                    "superpixel_extractor",
                    "params",
                    "nr_superpixels",
                ],
                [500, 1000],
            ),
            ParameterList(
                ["preprocess", "stages", "feature_extractor", "params", "architecture"],
                ["resnet18", "resnet34", "resnet50"],
            ),
        ],
    )
    PreprocessingExperiment(name="v0_few_superpixels",).generate(
        fixed=[
            Parameter(["preprocess", "params", "only_superpixel"], False),
            Parameter(
                ["preprocess", "stages", "stain_normalizer", "params", "target"],
                "ZT111_4_C_7_1",
            ),
        ],
        grid=[
            ParameterList(
                [
                    "preprocess",
                    "stages",
                    "superpixel_extractor",
                    "params",
                    "nr_superpixels",
                ],
                [100, 200, 300, 400, 600, 800],
            ),
            ParameterList(
                ["preprocess", "stages", "feature_extractor", "params", "architecture"],
                ["resnet18", "resnet34", "resnet50"],
            ),
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
                list(map(float, np.logspace(-3, -6, 20))),
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
                list(map(float, np.logspace(-2, -5, 10))),
            ),
        ]
    )
    PretrainingExperiment(name="batch_sizes").generate(
        sequential=[
            ParameterList(
                ["train", "params", "batch_size"],
                [1, 2, 4, 8, 16, 32, 64],
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
    PretrainingExperiment(name="basic_scratch_mobile").generate(
        fixed=[
            Parameter(
                ["train", "model", "pretrained"],
                False,
            ),
            Parameter(["train", "data", "train_fraction"], 0.75),
            Parameter(["train", "data", "training_slides"], None),
            Parameter(["train", "data", "validation_slides"], None),
            Parameter(["train", "model", "architecture"], "mobilenet_v2"),
        ],
        grid=[
            ParameterList(
                ["train", "params", "optimizer", "params", "lr"],
                [0.001, 0.0001, 0.00001],
            ),
            ParameterList(["train", "model", "width_mult"], [0.5, 1.0]),
            ParameterList(["train", "model", "dropout"], [0.0, 0.2, 0.3]),
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
    PretrainingExperiment(name="fine_tune_augmentations").generate(
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
    PretrainingExperiment(name="seperate_fine_tune_augmentations").generate(
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
