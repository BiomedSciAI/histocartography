import argparse
from pathlib import Path

import yaml


def get_lsf(
    config_name,
    queue="prod.med",
    cores=5,
    log_dir="/dataT/anv/logs",
    log_name="preprocessing",
    main_file_name="experiment",
    command="preprocess",
    nosave=False,
    subsample=None,
):
    return (
        f"#!/bin/bash\n\n"
        f"#BSUB -q {queue}\n"
        f"#BSUB -n {cores}\n\n"
        f"module purge\n"
        f"module load Miniconda3\n"
        f"source activate histocartography\n\n"
        f'#BSUB -J "{log_dir}/{log_name}"\n'
        f'#BSUB -o "{log_dir}/{log_name}"\n'
        f'#BSUB -e "{log_dir}/{log_name}.stderr"\n\n'
        f"python {main_file_name}.py {command} "
        f"--config {{PATH}}/{config_name}.yml "
        f"{'--nosave ' if nosave else ''}"
        f"{f'--subsample {subsample}' if subsample is not None else ''}"
        f"\n"
    )


def generate_performance_test(path: str, base: str):
    with open(base) as file:
        config: dict = yaml.load(file, Loader=yaml.FullLoader)

    job_name = "performance_test"
    job_id = 0
    subsample = 56
    for cores in [1]:
        for threads_per_core in [1, 2, 3, 4, 5, 6, 7, 8]:
            # Generate config
            new_config = config.copy()
            new_config["preprocess"]["params"]["cores"] = cores * threads_per_core
            new_config["preprocess"]["stages"]["superpixel_extractor"]["params"] = {
                "nr_superpixels": 1000,
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
