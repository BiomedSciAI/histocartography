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
    CPUPreprocessingExperiment(
        name="augmented_new_pretrained",
        queue="prod.med",
        base="config/augmented_preprocess.yml",
        path=PATH,
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
        path=PATH,
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
        path=PATH,
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
        path=PATH,
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
        path=PATH,
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
        path=PATH,
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
        name="grid_graph", base="config/augmented_preprocess.yml", path=PATH
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
        name="non_overlapping_graph", base="config/augmented_preprocess.yml", path=PATH
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
        "color_merged_low_no_overlap", base="config/merged_preprocess.yml", path=PATH
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
        "color_merged_med_no_overlap", base="config/merged_preprocess.yml", path=PATH
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
        "color_merged_high_no_overlap", base="config/merged_preprocess.yml", path=PATH
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
        "color_merged_very_high_no_overlap",
        base="config/merged_preprocess.yml",
        path=PATH,
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
    GPUPreprocessingExperiment(
        name="v10_no", base="config/augmented_preprocess.yml", path=PATH
    ).generate(
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
    GPUPreprocessingExperiment(
        name="v11_standard", base="config/new_feature.yml", path=PATH
    ).generate(
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
        ],
    )
    GPUPreprocessingExperiment(
        name="v11_less_context", base="config/new_feature.yml", path=PATH
    ).generate(
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
        ],
    )
    GPUPreprocessingExperiment(
        name="v11_more_finegrained",
        base="config/new_feature.yml",
        queue="prod.long",
        path=PATH,
    ).generate(
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
        ],
    )
    CPUPreprocessingExperiment(
        name="v11_standard_low", base="config/new_preprocess.yml", path=PATH
    ).generate(
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
                    [f"v11_mobilenet_low_{s}" for s in ["20x", "13x", "10x"]],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="v11_standard_med", base="config/new_preprocess.yml", path=PATH
    ).generate(
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
                    [f"v11_mobilenet_med_{s}" for s in ["20x", "13x", "10x"]],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="v11_standard_high", base="config/new_preprocess.yml", path=PATH
    ).generate(
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
                    [f"v11_mobilenet_high_{s}" for s in ["20x", "13x", "10x"]],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="v11_standard_very_high", base="config/new_preprocess.yml", path=PATH
    ).generate(
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
                    [f"v11_mobilenet_very_high_{s}" for s in ["20x", "13x", "10x"]],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="v11_standard_no", base="config/new_preprocess_no_merge.yml", path=PATH
    ).generate(
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
                    [f"v11_mobilenet_no_{s}" for s in ["20x", "13x", "10x"]],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="v11_standard_40x", base="config/new_preprocess.yml", path=PATH
    ).generate(
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
            Parameter(
                ["superpixel", "params", "nr_superpixels"],
                1200,
            ),
            Parameter(
                ["superpixel", "params", "compactness"],
                30,
            ),
            Parameter(
                ["feature_extraction", "params", "downsample_factor"],
                1,
            ),
        ],
        sequential=[
            [
                ParameterList(
                    ["superpixel", "params", "threshold"], [0.01, 0.02, 0.03, 0.04]
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [
                        f"v11_mobilenet_{s}_40x"
                        for s in ["low", "med", "high", "very_high"]
                    ],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="v11_standard_no_40x", base="config/new_preprocess_no_merge.yml", path=PATH
    ).generate(
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
            Parameter(
                ["superpixel", "params", "nr_superpixels"],
                1200,
            ),
            Parameter(
                ["superpixel", "params", "compactness"],
                30,
            ),
            Parameter(
                ["feature_extraction", "params", "downsample_factor"],
                1,
            ),
            Parameter(
                ["params", "link_directory"],
                "v11_mobilenet_no_40x",
            ),
        ],
    )
    CPUPreprocessingExperiment(
        name="v12_standard_low", base="config/new_preprocess.yml", path=PATH
    ).generate(
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
                    [700, 400, 250, 150],
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"],
                    [30, 30, 30, 30],
                ),
                ParameterList(
                    ["feature_extraction", "params", "downsample_factor"],
                    [1, 2, 3, 4],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [f"v12_mobilenet_low_{s}" for s in ["40x", "20x", "13x", "10x"]],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="v12_standard_med", base="config/new_preprocess.yml", path=PATH
    ).generate(
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
                    [700, 400, 250, 150],
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"],
                    [30, 30, 30, 30],
                ),
                ParameterList(
                    ["feature_extraction", "params", "downsample_factor"],
                    [1, 2, 3, 4],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [f"v12_mobilenet_med_{s}" for s in ["40x", "20x", "13x", "10x"]],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="v12_standard_high", base="config/new_preprocess.yml", path=PATH
    ).generate(
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
                    [700, 400, 250, 150],
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"],
                    [30, 30, 30, 30],
                ),
                ParameterList(
                    ["feature_extraction", "params", "downsample_factor"],
                    [1, 2, 3, 4],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [f"v12_mobilenet_high_{s}" for s in ["40x", "20x", "13x", "10x"]],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="v12_standard_very_high", base="config/new_preprocess.yml", path=PATH
    ).generate(
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
                    [700, 400, 250, 150],
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"],
                    [30, 30, 30, 30],
                ),
                ParameterList(
                    ["feature_extraction", "params", "downsample_factor"],
                    [1, 2, 3, 4],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [
                        f"v12_mobilenet_very_high_{s}"
                        for s in ["40x", "20x", "13x", "10x"]
                    ],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="v12_standard_no", base="config/new_preprocess_no_merge.yml", path=PATH
    ).generate(
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
                    [700, 400, 250, 150],
                ),
                ParameterList(
                    ["superpixel", "params", "compactness"],
                    [30, 30, 30, 30],
                ),
                ParameterList(
                    ["feature_extraction", "params", "downsample_factor"],
                    [1, 2, 3, 4],
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [f"v12_mobilenet_no_{s}" for s in ["40x", "20x", "13x", "10x"]],
                ),
            ]
        ],
    )

    # ETH
    StronglySupervisedGraphClassificationExperiment(
        name="multihop", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
    StronglySupervisedGraphClassificationExperiment(
        name="v9_color_merged", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
    StronglySupervisedGraphClassificationExperiment(
        name="v8_edge_merged", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
    WeaklySupervisedGraphClassificationExperiment(
        name="v10_old_image_level", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
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
    WeaklySupervisedGraphClassificationExperiment(
        name="v10_new_image_level", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
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
    WeaklySupervisedGraphClassificationExperiment(
        name="v10_image_level_dropout", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
        ],
        sequential=[
            ParameterList(
                ["train", "model", "graph_classifier_config", "input_dropout"],
                [0.0, 0.3, 0.5, 0.7],
            )
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
    WeaklySupervisedGraphClassificationExperiment(
        name="v10_gnn_layers", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                ["train", "model", "graph_classifier_config", "input_dropout"], 0.3
            ),
            Parameter(["train", "data", "augmentation_mode"], "graph"),
            Parameter(["train", "params", "optimizer", "params", "lr"], 1e-4),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {"class": "ExponentialLR", "params": {"gamma": 0.99}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "new_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
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
    WeaklySupervisedGraphClassificationExperiment(
        name="v10_augmentation", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                ["train", "model", "graph_classifier_config", "input_dropout"], 0.3
            ),
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
            ParameterList(
                ["train", "data", "augmentation_mode"], [None, "graph", "node"]
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
    WeaklySupervisedGraphClassificationExperiment(name="v10_lr", path=PATH).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                ["train", "model", "graph_classifier_config", "input_dropout"], 0.3
            ),
            Parameter(["train", "data", "image_labels_mode"], "new_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
            Parameter(
                ["train", "model", "gnn_config", "n_layers"],
                6,
            ),
            Parameter(["train", "data", "augmentation_mode"], "graph"),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "optimizer", "params", "lr"], [1e-3, 1e-4, 1e-5]
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
    WeaklySupervisedGraphClassificationExperiment(
        name="v11_old_image_level", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
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
    StronglySupervisedGraphClassificationExperiment(
        name="v11_pixel_level", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
    WeaklySupervisedGraphClassificationExperiment(
        name="v11_no_old_image_level", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
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
                    for level in ["no"]
                ],
            )
        ],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="v11_no_pixel_level", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                    f"outputs/v11_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "20x",
                        "13x",
                        "10x",
                    ]
                    for level in ["no"]
                ],
            )
        ],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="v11_old_image_level_longer", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                {"class": "ExponentialLR", "params": {"gamma": 0.995}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "original_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
            Parameter(["train", "params", "nr_epochs"], 1000),
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
    WeaklySupervisedGraphClassificationExperiment(
        name="v11_no_old_image_level_longer", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                {"class": "ExponentialLR", "params": {"gamma": 0.995}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "original_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
            Parameter(["train", "params", "nr_epochs"], 1000),
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
                    for level in ["no"]
                ],
            )
        ],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="v11_new_image_level_longer", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                {"class": "ExponentialLR", "params": {"gamma": 0.995}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "new_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
            Parameter(["train", "params", "nr_epochs"], 1000),
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
                    for level in ["no", "low", "med", "high", "very_high"]
                ],
            )
        ],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="v11_old_image_level_max", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                {"class": "ExponentialLR", "params": {"gamma": 0.995}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "original_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
            Parameter(["train", "params", "nr_epochs"], 1000),
        ],
        sequential=[
            ParameterList(
                ["train", "model", "gnn_config", "neighbor_pooling_type"], ["max"]
            )
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
                    for level in ["no", "low", "med", "high", "very_high"]
                ],
            )
        ],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="v11_old_image_level_gnn", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                {"class": "ExponentialLR", "params": {"gamma": 0.995}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "original_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
            Parameter(["train", "params", "nr_epochs"], 1000),
        ],
        sequential=[
            [
                ParameterList(
                    ["train", "model", "gnn_config", "hidden_dim"],
                    [16, 64],
                ),
                Parameter(
                    ["train", "model", "gnn_config", "output_dim"],
                    [16, 64],
                ),
            ]
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
                    for level in ["no", "low", "med", "high", "very_high"]
                ],
            )
        ],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="v11_old_image_level_longer", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                {"class": "ExponentialLR", "params": {"gamma": 0.995}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "original_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
            Parameter(["train", "params", "nr_epochs"], 1000),
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
    WeaklySupervisedGraphClassificationExperiment(
        name="v12_old_image_level_longer", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                {"class": "ExponentialLR", "params": {"gamma": 0.995}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "original_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
            Parameter(["train", "params", "nr_epochs"], 1000),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v12_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "40x",
                        "20x",
                        "13x",
                        "10x",
                    ]
                    for level in ["very_high", "high", "med", "low", "no"]
                ],
            )
        ],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="v12_new_image_level_longer", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                {"class": "ExponentialLR", "params": {"gamma": 0.995}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "new_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
            Parameter(["train", "params", "nr_epochs"], 1000),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v12_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "40x",
                        "20x",
                        "13x",
                        "10x",
                    ]
                    for level in ["very_high", "high", "med", "low", "no"]
                ],
            )
        ],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="v11_previous_pixel_level", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                6,
            ),
            Parameter(["train", "params", "optimizer", "params", "lr"], 3e-5),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v11_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "40x",
                        "20x",
                        "13x",
                        "10x",
                    ]
                    for level in ["no", "low", "med", "high", "very_high"]
                ],
            )
        ],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="v12_previous_pixel_level", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                6,
            ),
            Parameter(["train", "params", "optimizer", "params", "lr"], 3e-5),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v12_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "40x",
                        "20x",
                        "13x",
                        "10x",
                    ]
                    for level in ["no", "low", "med", "high", "very_high"]
                ],
            )
        ],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="v10_image_level_new", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                {"class": "ExponentialLR", "params": {"gamma": 0.995}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "new_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
            Parameter(["train", "params", "nr_epochs"], 1000),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v10_mobilenet_{level}_{magnification}_no_overlap"
                    for magnification in [
                        "40x",
                        "30x",
                        "20x",
                    ]
                    for level in ["no", "low", "med", "high", "very_high"]
                ],
            )
        ],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="v10_image_level_old", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                {"class": "ExponentialLR", "params": {"gamma": 0.995}},
            ),
            Parameter(["train", "data", "image_labels_mode"], "original_labels"),
            Parameter(["train", "data", "supervision", "mode"], "image_level"),
            Parameter(["train", "params", "nr_epochs"], 1000),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v10_mobilenet_{level}_{magnification}_no_overlap"
                    for magnification in [
                        "40x",
                        "30x",
                        "20x",
                    ]
                    for level in ["no", "low", "med", "high", "very_high"]
                ],
            )
        ],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="v10_pixel_level", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "use_augmentation_dataset"], True),
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
                6,
            ),
            Parameter(["train", "params", "optimizer", "params", "lr"], 3e-5),
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v10_mobilenet_{level}_{magnification}_no_overlap"
                    for magnification in [
                        "40x",
                        "30x",
                        "20x",
                    ]
                    for level in ["no", "low", "med", "high", "very_high"]
                ],
            )
        ],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="v12_combine_13x", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "graph"),
        ],
        sequential=[
            ParameterList(["train", "params", "loss", "node_weight"], [0.2, 0.5, 0.8])
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v12_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "13x",
                    ]
                    for level in ["med", "high"]
                ],
            )
        ],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="v11_med_13x_best_pixel", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x",
            ),
            Parameter(["train", "data", "augmentation_mode"], "graph"),
            Parameter(["train", "model", "gnn_config", "n_layers"], 6),
            Parameter(["train", "model", "gnn_config", "dropout"], 0.5),
            Parameter(["train", "model", "node_classifier_config", "n_layers"], 2),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], True
            ),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "loss", "node_weight"], [0.1, 0.3, 0.5, 0.7, 0.9]
            )
        ],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="v10_med_13x_best_image", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v10_mobilenet_med_30x_no_overlap",
            ),
            Parameter(["train", "data", "augmentation_mode"], "graph"),
            Parameter(["train", "model", "gnn_config", "dropout"], 0.5),
            Parameter(["train", "model", "node_classifier_config", "n_layers"], 2),
            Parameter(
                ["train", "model", "graph_classifier_config", "input_dropout"], 0.5
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], True
            ),
            Parameter(["train", "params", "optimizer", "params", "lr"], 0.0001),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {"class": "ExponentialLR", "params": {"gamma": 0.995}},
            ),
            Parameter(["train", "params", "nr_epochs"], 1000),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "loss", "node_weight"], [0.1, 0.3, 0.5, 0.7, 0.9]
            )
        ],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="v12_ps_combine_13x", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "graph"),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
        ],
        sequential=[
            ParameterList(["train", "params", "loss", "node_weight"], [0.2, 0.5, 0.8])
        ],
        grid=[
            ParameterList(
                ["train", "data", "graph_directory"],
                [
                    f"outputs/v12_mobilenet_{level}_{magnification}"
                    for magnification in [
                        "13x",
                    ]
                    for level in ["med", "high"]
                ],
            )
        ],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="v11_ps_med_13x_best_pixel", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
            Parameter(["train", "data", "augmentation_mode"], "graph"),
            Parameter(["train", "model", "gnn_config", "n_layers"], 6),
            Parameter(["train", "model", "gnn_config", "dropout"], 0.5),
            Parameter(["train", "model", "node_classifier_config", "n_layers"], 2),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], True
            ),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "loss", "node_weight"], [0.1, 0.3, 0.5, 0.7, 0.9]
            )
        ],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="v10_ps_med_13x_best_image", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v10_mobilenet_med_30x_no_overlap",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
            Parameter(["train", "data", "augmentation_mode"], "graph"),
            Parameter(["train", "model", "gnn_config", "dropout"], 0.5),
            Parameter(["train", "model", "node_classifier_config", "n_layers"], 2),
            Parameter(
                ["train", "model", "graph_classifier_config", "input_dropout"], 0.5
            ),
            Parameter(
                ["train", "model", "node_classifier_config", "seperate_heads"], True
            ),
            Parameter(["train", "params", "optimizer", "params", "lr"], 0.0001),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {"class": "ExponentialLR", "params": {"gamma": 0.995}},
            ),
            Parameter(["train", "params", "nr_epochs"], 1000),
        ],
        sequential=[
            ParameterList(
                ["train", "params", "loss", "node_weight"], [0.1, 0.3, 0.5, 0.7, 0.9]
            )
        ],
    )

    # Pretraining
    PretrainingExperiment(name="baseline", queue="prod.long", path=PATH).generate(
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
    PretrainingExperiment(name="optimizer_long", queue="prod.long", path=PATH).generate(
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
    PretrainingExperiment(
        name="batch_sizes_long", queue="prod.long", path=PATH
    ).generate(
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
    PretrainingExperiment(name="step_lr", path=PATH).generate(
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
    PretrainingExperiment(name="exponential_lr", path=PATH).generate(
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
    PretrainingExperiment(name="drop_patches", path=PATH).generate(
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
    PretrainingExperiment(name="drop_patches_on_val", path=PATH).generate(
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
    PretrainingExperiment(name="drop_unlabelled", path=PATH).generate(
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
    PretrainingExperiment(name="balanced_batches", path=PATH).generate(
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
    PretrainingExperiment(name="encoder_pretraining", path=PATH).generate(
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

    CNNTestingExperiment(name="exponential_lr_decay", path=PATH).generate(
        fixed=[
            Parameter(
                ["test", "model", "architecture"],
                "s3://mlflow/633/02906fe539444b13a76d39d4a0dfbb6f/artifacts/best.valid.MultiLabelBalancedAccuracy",
            )
        ],
        sequential=[ParameterList(["test", "params", "overlap"], [150, 175, 200, 210])],
    )
    CNNTestingExperiment(name="step_lr_decay", path=PATH).generate(
        fixed=[
            Parameter(
                ["test", "model", "architecture"],
                "s3://mlflow/633/c62233eed1574d2ca2d9b8ee51b83ffc/artifacts/best.valid.MultiLabelBalancedAccuracy",
            )
        ],
        sequential=[ParameterList(["test", "params", "overlap"], [150, 175, 200, 210])],
    )
    CNNTestingExperiment(name="batch_size_and_sgd", path=PATH).generate(
        fixed=[
            Parameter(
                ["test", "model", "architecture"],
                "s3://mlflow/633/19a9b40d174f40c4b217ddf84eb63e3b/artifacts/best.valid.MultiLabelBalancedAccuracy",
            )
        ],
        sequential=[ParameterList(["test", "params", "overlap"], [150, 175, 200, 210])],
    )
    CNNTestingExperiment(name="various", path=PATH).generate(
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
    CNNTestingExperiment(name="thresholds", path=PATH).generate(
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
    CNNTestingExperiment(name="normal_loader", path=PATH).generate(
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
    CNNTestingExperiment(name="balanced_loader", path=PATH).generate(
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
    CNNTestingExperiment(name="classifier_pretrain", path=PATH).generate(
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
    CNNTestingExperiment(name="rerun_all", path=PATH).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "rep_eth",
                            "rep_eth",
                            "rep_eth",
                            "rep_best_mean_iou",
                            "rep_best_mean_iou",
                            "rep_best_mean_iou",
                            "rep_good_mean_iou",
                            "rep_good_mean_iou",
                            "rep_good_mean_iou",
                            "rep_best_kappa1",
                            "rep_best_kappa1",
                            "rep_best_kappa1",
                            "rep_best_kappa2",
                            "rep_best_kappa2",
                            "rep_best_kappa2",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        f"s3://mlflow/633/{run}/artifacts/best.valid.MultiLabelBalancedAccuracy"
                        for run in [
                            "f3847e42206246eba25d818fae0f7135",
                            "652cde8de3c34da4b4ba3eaaa90e37a4",
                            "d06cc995335d4c4f9153f390080e8249",
                            "2d1a8a2150be4aa4b1db35198f63326c",
                            "8c25ef33f3b84a4f8b5dfb27a02b2b53",
                            "cc9e1b7a8efb4ff2954e5302393e1cde",
                            "96a174efeab3459f8f7b944fe304fdb4",
                            "271a7d5fd31a4ea3a657d747db65d924",
                            "0c7c1b4389d84daf8086ad4c6afde5ea",
                            "74366b6f8ec949108ae5ecd185be18f6",
                            "a11e92754bfa45a3b0fd288ece0f7c93",
                            "a1a4d4ea838e49e99312c0cb536652a2",
                            "734fc44a6db048f5a081c33d0ba07428",
                            "b717a2fe84394895b77e43e032ee0168",
                            "fd4be6f152384f55a34dbfd088936f38",
                        ]
                    ],
                ),
            ]
        ],
        grid=[
            ParameterList(
                ["test", "params", "inference_mode"], ["patch_based", "hacky"]
            )
        ],
    )
    GNNTestingExperiment(
        name="rerun_failed", base="config/final_pixel.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_rep_best_meaniou",
                            "tissue_rep_best_meaniou",
                            "tissue_rep_best_meaniou",
                            "tissue_rep_good_meaniou",
                            "tissue_rep_good_meaniou",
                            "tissue_rep_good_meaniou",
                            "tissue_rep_best_kappa",
                            "tissue_rep_best_kappa",
                            "tissue_rep_best_kappa",
                            "tissue_rep_good_kappa",
                            "tissue_rep_good_kappa",
                            "tissue_rep_good_kappa",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        f"s3://mlflow/645/{run}/artifacts/best.valid.node.segmentation.MeanIoU"
                        for run in [
                            "4021ebae8da042e3802cbfab5bee7a24",
                            "6cf34a13418e45ee87d6074f8dcbdb46",
                            "89a0e82b66ea4dec8342bd9faea12266",
                            "22ea6ccf4853411bbaba3de668a65cff",
                            "7f1f177907144e0d891633f553fe5927",
                            "3a3523d8ef8e4227acffc32618b90f2f",
                            "525aca9ccf3c4cd69b7eee34ad636319",
                            "0a5d9ae21d5b4363a49edb7853d4c272",
                            "4f55632690cc411794f6c1c2d6d1bbbf",
                            "a0f9dd09b47f4a0caa5ac5801a98a480",
                            "46d614ab20744e3f97f9f68be7d89a99",
                            "819a33e7136440cb934ba406cc4cd318",
                        ]
                    ],
                ),
            ]
        ],
    )

    # FINAL WEAKLY SUPERVISED RESULTS
    GraphClassifierExperiment(
        name="rep_ws_v11_1", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"], f"outputs/v11_mobilenet_high_40x"
            )
        ],
        repetitions=3,
    )
    GraphClassifierExperiment(
        name="rep_ws_v11_2", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"], f"outputs/v11_mobilenet_high_20x"
            )
        ],
        repetitions=3,
    )
    GraphClassifierExperiment(
        name="rep_ws_v11_3", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"], f"outputs/v11_mobilenet_high_13x"
            )
        ],
        repetitions=3,
    )
    GraphClassifierExperiment(
        name="rep_ws_v11_4", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"], f"outputs/v11_mobilenet_high_10x"
            )
        ],
        repetitions=3,
    )
    GraphClassifierExperiment(
        name="rep_ws_v12_1", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"], f"outputs/v12_mobilenet_high_40x"
            )
        ],
        repetitions=3,
    )
    GraphClassifierExperiment(
        name="rep_ws_v12_2", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"], f"outputs/v12_mobilenet_high_20x"
            )
        ],
        repetitions=3,
    )
    GraphClassifierExperiment(
        name="rep_ws_v12_3", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"], f"outputs/v12_mobilenet_high_13x"
            )
        ],
        repetitions=3,
    )
    GraphClassifierExperiment(
        name="rep_ws_v12_4", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"], f"outputs/v12_mobilenet_high_10x"
            )
        ],
        repetitions=3,
    )

    # FINAL CNN (ETH Parameters)
    PretrainingExperiment(
        name="rep_eth", base="config/final_cnn_eth.yml", queue="prod.long", path=PATH
    ).generate(repetitions=3)
    PretrainingExperiment(
        name="rep_best_meaniou",
        base="config/final_cnn.yml",
        queue="prod.long",
        path=PATH,
    ).generate(repetitions=3)
    PretrainingExperiment(
        name="rep_good_meaniou",
        base="config/final_cnn2.yml",
        queue="prod.long",
        path=PATH,
    ).generate(repetitions=3)
    PretrainingExperiment(
        name="rep_best_kappa1",
        base="config/final_cnn3.yml",
        queue="prod.long",
        path=PATH,
    ).generate(repetitions=3)
    PretrainingExperiment(
        name="rep_best_kappa2",
        base="config/final_cnn4.yml",
        queue="prod.long",
        path=PATH,
    ).generate(repetitions=3)

    # FINAL PIXEL LEVEL
    StronglySupervisedGraphClassificationExperiment(
        name="rep_best_meaniou", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x",
            )
        ],
        repetitions=3,
    )
    StronglySupervisedGraphClassificationExperiment(
        name="rep_good_meaniou", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_very_high_10x",
            )
        ],
        repetitions=3,
    )
    StronglySupervisedGraphClassificationExperiment(
        name="rep_best_kappa", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_low_20x",
            )
        ],
        repetitions=3,
    )
    StronglySupervisedGraphClassificationExperiment(
        name="rep_good_kappa", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_no_13x",
            )
        ],
        repetitions=3,
    )

    # FINAL TMA LEVEL
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_best_meaniou", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v10_mobilenet_med_30x_no_overlap",
            )
        ],
        repetitions=3,
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_good_meaniou", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_no_10x",
            )
        ],
        repetitions=3,
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_best_kappa", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_low_13x",
            )
        ],
        repetitions=3,
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_good_kappa", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_no_10x",
            )
        ],
        repetitions=3,
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_ps_best_meaniou", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v10_mobilenet_med_30x_no_overlap",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
        ],
        repetitions=3,
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_ps_good_meaniou", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_no_10x",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
        ],
        repetitions=3,
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_ps_best_kappa", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_low_13x",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
        ],
        repetitions=3,
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_ps_good_kappa", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_no_10x",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
        ],
        repetitions=3,
    )
