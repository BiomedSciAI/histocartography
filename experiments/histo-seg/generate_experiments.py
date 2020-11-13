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


def preprocess_nr_superpixels(path: str, base: str):
    with open(base) as file:
        config: dict = yaml.load(file, Loader=yaml.FullLoader)

    job_name = "preprocessing_superpixels"
    job_id = 0
    cores = 5
    for nr_superpixels in [100, 250, 500, 1000, 2000, 4000, 8000]:
        # Generate config
        new_config = config.copy()
        new_config["preprocess"]["params"]["cores"] = cores * 6
        new_config["preprocess"]["stages"]["superpixel_extractor"]["params"] = {
            "nr_superpixels": nr_superpixels,
        }

        # Generate lsf file
        lsf_content = get_lsf(
            config_name=f"job{job_id}",
            queue="prod.med",
            cores=cores,
            log_name=f"{job_name}{job_id}",
            main_file_name="preprocess",
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
    preprocess_nr_superpixels(path=args.path, base=args.base)
    Experiment(
        name="preproessing_handcrafted",
        cores=8,
        gpus=0,
        main_file="preprocess",
        disable_multithreading=True,
    ).generate(
        fixed=[
            Parameter(["preprocess", "params", "cores"], 8 * 7),
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
    Experiment(
        name="preproessing_resnet34",
        cores=8,
        gpus=0,
        main_file="preprocess",
        disable_multithreading=True,
    ).generate(
        fixed=[
            Parameter(["preprocess", "params", "cores"], 8 * 7),
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
    Experiment(
        name="preproessing_lowres",
        cores=6,
        gpus=0,
        main_file="preprocess",
        disable_multithreading=True,
    ).generate(
        fixed=[
            Parameter(["preprocess", "params", "cores"], 6 * 7),
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

    # ETH
    Experiment(name="train_basic_search").generate(
        sequential=[
            ParameterList(
                ["train", "params", "optimizer", "params", "lr"],
                [0.0125, 0.0025, 0.0005, 0.0001, 0.00002],
            ),
            ParameterList(
                ["train", "model", "gnn_config", "n_layers"], [2, 3, 4, 5, 6, 7, 8]
            ),
            ParameterList(["train", "data", "patch_size"], [1000, 2000, 3000]),
        ],
    )
    Experiment(name="node_stochasticity").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "node_stochasticity", "fix": "add_centroid_features"},
            )
        ],
        sequential=[
            ParameterList(
                ["train", "params", "loss", "node", "params", "drop_probability"],
                [0.99, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1],
            ),
        ],
    )
    Experiment(name="node_loss_weight").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "node_loss_weight", "fix": "add_centroid_features"},
            )
        ],
        sequential=[
            ParameterList(
                ["train", "params", "loss", "node_weight"],
                [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            ),
        ],
    )
    Experiment(name="batch_sizes").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "batch_size", "fix": "add_centroid_features"},
            )
        ],
        sequential=[
            ParameterList(
                ["train", "params", "batch_size"],
                [1, 2, 4, 8, 16, 32, 64],
            ),
        ],
    )
    Experiment(name="learning_rates").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "learning_rate", "fix": "add_centroid_features"},
            )
        ],
        sequential=[
            ParameterList(
                ["train", "params", "optimizer", "params", "lr"],
                list(map(float, np.logspace(-3, -6, 20))),
            ),
        ],
    )
    Experiment(name="nr_superpixels").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "nr_superpixels", "fix": "add_centroid_features"},
            )
        ],
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
    Experiment(name="crop_augmentation").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "patch_size", "fix": "add_centroid_features"},
            )
        ],
        sequential=[
            ParameterList(
                ["train", "data", "patch_size"],
                [1900, 2200, 2800, 3100],
            ),
        ],
    )
    Experiment(name="gnn_parameters").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {
                    "grid_search": "various_gnn_parameters",
                    "fix": "add_centroid_features",
                },
            )
        ],
        grid=[
            ParameterList(
                ["train", "model", "gnn_config", "agg_operator"],
                ["none", "concat", "lstm"],
            ),
            ParameterList(["train", "model", "gnn_config", "n_layers"], [2, 4, 6]),
            ParameterList(["train", "model", "gnn_config", "hidden_dim"], [32, 64]),
        ],
    )
    Experiment(name="centroid_features").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "centroid_features", "fix": "add_centroid_features"},
            )
        ],
        sequential=[
            [
                ParameterList(["train", "data", "centroid_features"], ["only"]),
                ParameterList(["train", "model", "gnn_config", "input_dim"], [2]),
            ]
        ],
    )
    Experiment(name="lr_gradient_norm").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "learning_rate", "fix": "add_centroid_features"},
            ),
            Parameter(["train", "params", "clip_gradient_norm"], 5.0),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "optimizer", "params", "lr"],
                list(map(float, np.logspace(-2, -5, 10))),
            ),
        ],
    )
    Experiment(name="sanity_check").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"fix": "sanity_check"},
            )
        ],
    )
    Experiment(name="tiny_centroid_networks").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "tiny_centroid_networks"},
            ),
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
    Experiment(name="small_networks").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "small_networks"},
            ),
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
    Experiment(name="baseline").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "baseline"},
            ),
        ],
        sequential=[
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers" : 2, "hidden_dim" : 16}]
            )
        ]
    )
    Experiment(name="gnn_agg").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "aggregator"},
            ),
        ],
        grid=[
            ParameterList(
                ["train", "model", "gnn_config", "neighbor_pooling_type"],
                ["sum", "min", "max"],
            ),
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers" : 2, "hidden_dim" : 16}]
            )
        ],
    )
    Experiment(name="normalize").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "normalize"},
            )
        ],
        grid=[
            ParameterList(["train", "data", "normalize_features"], [True]),
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers" : 2, "hidden_dim" : 16}]
            )
        ],
    )
    Experiment(name="fold_sanity_check").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "fold_sanity_check"},
            )
        ],
        sequential=[
            ParameterList(["train", "data", "training_slides"], [
                [76, 199, 204], [111, 76, 199]
            ]),
            ParameterList(["train", "data", "validation_slides"], [
                [111], [204]
            ])
        ],
    )
    Experiment(name="handcrafted_features").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "handcrafted_features"},
            ),
            Parameter(
                ["train", "data", "normalize_features"],
                True,
            ),
            Parameter(
                ["train", "model", "gnn_config", "input_dim"],
                59
            )
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                ["outputs/hc_500px", "outputs/hc_1000px", "outputs/hc_2000px"]
            ),
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers" : 2, "hidden_dim" : 16}]
            )
        ]
    )
    Experiment(name="resnet34").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "resnet34"},
            ),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                ["outputs/resnet34_500px", "outputs/resnet34_1000px", "outputs/resnet34_2000px"]
            ),
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers" : 2, "hidden_dim" : 16}]
            )
        ]
    )
    Experiment(name="downsampled_150").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "downsampled_150"},
            ),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                ["outputs/resnet34_500px_1.5x", "outputs/resnet34_1000px_1.5x"]
            ),
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers" : 2, "hidden_dim" : 16}]
            )
        ]
    )
    Experiment(name="downsampled_200").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "downsampled_200"},
            ),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                ["outputs/resnet34_500px_2x", "outputs/resnet34_1000px_2x"]
            ),
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers" : 2, "hidden_dim" : 16}]
            )
        ]
    )
    Experiment(name="downsampled_300").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "downsampled_300"},
            ),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                ["outputs/resnet34_500px_3x", "outputs/resnet34_1000px_3x"]
            ),
            ParameterList(
                ["train", "model", "node_classifier_config"],
                [None, {"n_layers" : 2, "hidden_dim" : 16}]
            )
        ]
    )

    # MNIST
    Experiment(name="mnist_batch_sizes", base="mnist.yml").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "batch_size"},
            ),
            Parameter(
                ["train", "params", "nr_epochs"],
                500
            )
        ],
        sequential=[
            ParameterList(
                ["train", "params", "batch_size"],
                [8, 16, 32, 64, 128],
            ),
        ],
    )
    Experiment(name="mnist_learning_rates", base="mnist.yml").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "learning_rate"},
            ),
            Parameter(
                ["train", "params", "nr_epochs"],
                500
            )
        ],
        sequential=[
            ParameterList(
                ["train", "params", "optimizer", "params", "lr"],
                list(map(float, np.logspace(-3, -6, 10)))[5:],
            ),
        ],
    )
    Experiment(name="mnist_centroid_features", base="mnist.yml").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "centroid_features"},
            )
        ],
        sequential=[
            [
                ParameterList(
                    ["train", "data", "centroid_features"], ["no", "only"]
                ),
                ParameterList(
                    ["train", "model", "gnn_config", "input_dim"], [57, 2]
                ),
            ],
        ],
    ),
    Experiment(name="mnist_no_normalizer", base="mnist.yml").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "no_normalizer"},
            )
        ],
        sequential=[
            ParameterList(["train", "data", "normalize_features"], [True, False]),
        ],
    ),
    Experiment(name="mnist_gnn_config", base="mnist.yml").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "gnn_config"},
            )
        ],
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
    Experiment(name="mnist_batch_sizes_cpu", base="mnist.yml").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "batch_size"},
            ),
            Parameter(
                ["train", "params", "nr_epochs"],
                500
            )
        ],
        sequential=[
            ParameterList(
                ["train", "params", "batch_size"],
                [8, 16, 32, 64, 128],
            ),
        ],
    )
    Experiment(name="mnist_agg", base="mnist.yml").generate(
        fixed=[
            Parameter(
                ["train", "params", "experiment_tags"],
                {"grid_search": "aggregator"},
            ),
        ],
        sequential=[
            ParameterList(
                ["train", "model", "gnn_config", "neighbor_pooling_type"],
                ["mean", "sum", "min", "max"],
            ),
        ],
    )