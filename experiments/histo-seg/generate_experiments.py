import argparse
import copy
from pathlib import Path

import yaml


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
        f'export PYTHONPATH="$PWD/../../:{{$PYTHONPATH}}"\n'
        f"{'OMP_NUM_THREADS=1' if disable_multithreading else ''}"
        f"python {main_file_name}.py "
        f"--config {{PATH}}/{config_name}.yml "
        f"{'--nosave ' if nosave else ''}"
        f"{f'--subsample {subsample}' if subsample is not None else ''}"
        f"\n"
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


def tiny_grid_search(path: str, base: str):
    with open(base) as file:
        config: dict = yaml.load(file, Loader=yaml.FullLoader)

    job_name = "train_basic_search"
    job_id = 0

    def schedule_job(new_config):
        # Generate lsf file
        lsf_content = get_lsf(
            config_name=f"job{job_id}",
            queue="prod.med",
            cores=1,
            gpus=1,
            log_name=f"{job_name}{job_id}",
            main_file_name="train",
        )

        # Write files
        target_directory = Path(path) / job_name
        if not target_directory.exists():
            target_directory.mkdir()
        with open(target_directory / f"job{job_id}.lsf", "w") as file:
            file.write(lsf_content)
        with open(target_directory / f"job{job_id}.yml", "w") as file:
            yaml.dump(new_config, file)

    for lr in [0.0125, 0.0025, 0.0005, 0.0001, 0.00002]:
        new_config = copy.deepcopy(config)
        new_config["train"]["params"]["optimizer"]["params"]["lr"] = lr
        schedule_job(new_config=new_config)
        job_id += 1
    for n_layers in [2, 3, 4, 5, 6, 7, 8]:
        new_config = copy.deepcopy(config)
        new_config["train"]["model"]["gnn_config"]["n_layers"] = n_layers
        schedule_job(new_config=new_config)
        job_id += 1
    for patch_size in [1000, 2000, 3000]:
        new_config = copy.deepcopy(config)
        new_config["train"]["data"]["patch_size"] = patch_size
        schedule_job(new_config=new_config)
        job_id += 1


def preprocess_nr_superpixels(path: str, base: str):
    with open(base) as file:
        config: dict = yaml.load(file, Loader=yaml.FullLoader)

    job_name = "preprocessing_superpixels"
    job_id = 0
    cores = 5
    for nr_superpixels in [100, 250, 500, 1000, 4000, 8000]:
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

    generate_performance_test(path=args.path, base=args.base)
    generate_upper_bounds(path=args.path, base=args.base)
    tiny_grid_search(path=args.path, base=args.base)
    preprocess_nr_superpixels(path=args.path, base=args.base)
