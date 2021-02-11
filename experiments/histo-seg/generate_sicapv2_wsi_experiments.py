import argparse
from pathlib import Path

import numpy as np

from experiment import (
    CPUPreprocessingExperiment,
    GPUPreprocessingExperiment,
    Parameter,
    ParameterList,
    SemiSupervisedGraphClassificationExperiment,
    StronglySupervisedGraphClassificationExperiment,
    WeaklySupervisedGraphClassificationExperiment,
    PATH,
    BASE
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="/Users/anv/Documents/experiment_configs"
    )
    parser.add_argument("--base", type=str, default="config/default_strong.yml")
    args = parser.parse_args()

    PATH = args.path
    BASE = str(Path("config") / args.base)

    # Preprocessing
    GPUPreprocessingExperiment(
        name="sicapv2_wsi_feat",
        base="config/feat_sicap_wsi.yml",
        queue="prod.p9",
        workers=24,
        path=PATH
    ).generate()
    CPUPreprocessingExperiment(
        name="sicapv2_wsi", base="config/preprocessing_sicap_wsi.yml", cores=4, path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["params", "link_directory"],
                "v0_low_4000",
            ),
        ]
    )

    # SiCAPv2 WSI dataset
    StronglySupervisedGraphClassificationExperiment(
        name="best_strong", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])])
    WeaklySupervisedGraphClassificationExperiment(
        name="best_weak", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])])
