import numpy as np
from pathlib import Path
import argparse
from experiment import *

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
    CPUPreprocessingExperiment(name="sicapv2_wsi_stain_norm", base="config/stain_norm_sicap_wsi.yml").generate()
    GPUPreprocessingExperiment(name="sicapv2_wsi_feat", base="config/feat_sicap_wsi.yml", queue="prod.p9", workers=24).generate()
    CPUPreprocessingExperiment(name="sicapv2_wsi", base="config/preprocessing_sicap_wsi.yml", cores=4).generate(
        fixed=[
            Parameter(
                ["params", "link_directory"],
                "v0_low_4000",
            ),
        ]
    )

    # SiCAPv2 WSI dataset
    StronglySupervisedGraphClassificationExperiment(name="best_strong", base="config/sicapv2_wsi_strong.yml").generate(
        grid=[
            ParameterList(["train", "data", "fold"], [1, 2, 3, 4])
        ]
    )
    WeaklySupervisedGraphClassificationExperiment(name="best_weak", base="config/sicapv2_wsi_weak.yml").generate(
        grid=[
            ParameterList(["train", "data", "fold"], [1, 2, 3, 4])
        ]
    )
