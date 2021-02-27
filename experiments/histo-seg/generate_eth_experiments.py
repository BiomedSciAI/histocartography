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
    GNNTestingExperiment(
        name="eth_rerun_tissue_base", base="config/final_pixel.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_rep_base",
                            "tissue_rep_base",
                            "tissue_rep_base",
                            "tissue_rep_base",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        f"s3://mlflow/645/{run}/artifacts/best.valid.node.segmentation.fF1Score"
                        for run in [
                            "940b1e83e0dc43e2bac552535d9b697f",
                            "927dcfe79495455887da10b6efb4349c",
                            "39d32a7ccbfb4c9fab8d0eca7ba35889",
                            "29c2922008ce4a5680823e6521fb243e",
                        ]
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="eth_rerun_tissue_lr", base="config/final_pixel.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_rep_lr_normal",
                            "tissue_rep_lr_normal",
                            "tissue_rep_lr_normal",
                            "tissue_rep_lr_normal",
                            "tissue_rep_lr_high",
                            "tissue_rep_lr_high",
                            "tissue_rep_lr_high",
                            "tissue_rep_lr_high",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        f"s3://mlflow/645/{run}/artifacts/best.valid.node.segmentation.fF1Score"
                        for run in [
                            "4a8d021f72d84695ac67a7895bd32992",
                            "5339c885ed5244d68efb89dc0882bb28",
                            "89e0b32c52ba44ec93084fbb20b634ca",
                            "e8725f643d784f6688c6e1a503d2099d",
                            "66bc564f9cd14ce3a1788f1dbf03eacc",
                            "a8bf7412ace54e9385384864ca9e3c83",
                            "89139d61e31849f9859f0846640f5748",
                            "ecc8aadc87934e9c88c206a232dcae8b",
                        ]
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="eth_rerun_image_lr", base="config/final_pixel.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "image_rep_lr_normal",
                            "image_rep_lr_normal",
                            "image_rep_lr_normal",
                            "image_rep_lr_normal",
                            "image_rep_lr_high",
                            "image_rep_lr_high",
                            "image_rep_lr_high",
                            "image_rep_lr_high",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        f"s3://mlflow/644/{run}/artifacts/best.valid.graph.segmentation.fF1Score"
                        for run in [
                            "09f883a406b644548b55a4bf96b81531",
                            "3125ac2bc7224701b7a0af20341661aa",
                            "a76850aa255a46d0a808963e09ecf781",
                            "4c8c64dc133243d9bc5731a641a6fb67",
                            "86667804397040af9921eb3620c38ba4",
                            "fa5408c6c33a4e5b833b87f48f1b2ad2",
                            "040e738edf8e494aa0a6c18863ae2cfc",
                            "b152eb866cac4defb394979be1df24f1",
                        ]
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="eth_image_big_rerun", base="config/final_weak.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "image_rep_lr_high",
                            "image_rep_lr_high",
                            "image_rep_lr_high",
                            "image_rep_lr_high",
                            "image_rep_lr_normal",
                            "image_rep_lr_normal",
                            "image_rep_lr_normal",
                            "image_rep_lr_normal",
                            "image_rep_weighted_ps_best_kappa",
                            "image_rep_weighted_ps_best_kappa",
                            "image_rep_weighted_ps_best_kappa",
                            "image_rep_weighted_ps_best_kappa",
                            "image_rep_weighted_ps_best_meaniou",
                            "image_rep_weighted_ps_best_meaniou",
                            "image_rep_weighted_ps_best_meaniou",
                            "image_rep_weighted_ps_best_meaniou",
                            "image_rep_fold_ps_good_meaniou",
                            "image_rep_fold_ps_good_meaniou",
                            "image_rep_fold_ps_good_meaniou",
                            "image_rep_fold_ps_good_meaniou",
                            "image_rep_fold_ps_good_kappa",
                            "image_rep_fold_ps_good_kappa",
                            "image_rep_fold_ps_good_kappa",
                            "image_rep_fold_ps_good_kappa",
                            "image_rep_fold_ps_best_meaniou",
                            "image_rep_fold_ps_best_meaniou",
                            "image_rep_fold_ps_best_meaniou",
                            "image_rep_fold_ps_best_meaniou",
                            "image_rep_fold_ps_best_kappa",
                            "image_rep_fold_ps_best_kappa",
                            "image_rep_fold_ps_best_kappa",
                            "image_rep_fold_ps_best_kappa",
                            "image_rep_fold_good_meaniou",
                            "image_rep_fold_good_meaniou",
                            "image_rep_fold_good_meaniou",
                            "image_rep_fold_good_meaniou",
                            "image_rep_fold_good_kappa",
                            "image_rep_fold_good_kappa",
                            "image_rep_fold_good_kappa",
                            "image_rep_fold_good_kappa",
                            "image_rep_fold_best_meaniou",
                            "image_rep_fold_best_meaniou",
                            "image_rep_fold_best_kappa",
                            "image_rep_fold_best_meaniou",
                            "image_rep_fold_best_kappa",
                            "image_rep_fold_best_meaniou",
                            "image_rep_fold_best_kappa",
                            "image_rep_fold_best_kappa",
                            "image_rep_ps_good_kappa",
                            "image_rep_ps_good_kappa",
                            "image_rep_ps_good_kappa",
                            "image_rep_ps_good_meaniou",
                            "image_rep_ps_good_meaniou",
                            "image_rep_ps_good_meaniou",
                            "image_rep_ps_best_meaniou",
                            "image_rep_ps_best_kappa",
                            "image_rep_ps_best_kappa",
                            "image_rep_ps_best_meaniou",
                            "image_rep_ps_best_kappa",
                            "image_rep_ps_best_meaniou",
                            "image_rep_good_kappa",
                            "image_rep_good_kappa",
                            "image_rep_good_kappa",
                            "image_rep_good_meaniou",
                            "image_rep_good_meaniou",
                            "image_rep_good_meaniou",
                            "image_rep_best_kappa",
                            "image_rep_best_kappa",
                            "image_rep_best_kappa",
                            "image_rep_best_meaniou",
                            "image_rep_best_meaniou",
                            "image_rep_best_meaniou",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/644/040e738edf8e494aa0a6c18863ae2cfc/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/b152eb866cac4defb394979be1df24f1/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/fa5408c6c33a4e5b833b87f48f1b2ad2/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/86667804397040af9921eb3620c38ba4/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/3125ac2bc7224701b7a0af20341661aa/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/09f883a406b644548b55a4bf96b81531/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/4c8c64dc133243d9bc5731a641a6fb67/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/a76850aa255a46d0a808963e09ecf781/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/a4e2cabee9444a2fbc805650a9f12c1a/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/21163b9f6b704117905da511e426e7df/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/6db533ad00a04c27ae3bbd6339cb412e/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/c6acb5dcb1804b6e8cb51a52456e565b/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/351b49cb2e4f4503a6de20757d65c308/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/ed7a3a42986b4eeeaeaed0c8e16ed37a/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/4396cfe73c654e9ead2e8aa1bf393d29/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/2b6224b8e66e4212b2059e47c2317db4/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/472ed41bf02845dfbaeac5eeb1db47bd/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/a35c2f7ecde647398501fd1895719728/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/bce23978c6754dcc85e3510e6709c9f2/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/3ee985c0098c495fa3444f79fa8b8f74/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/86fd3f462ed74a21b45693a0e35ad4dc/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/a41c4fe6fe584ce6b5fe2c133d4aaaa4/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/9863942714d0430e9c5a2549a3db436d/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/fa67a5f6f51a41bea6e8774fe1f15107/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/52a4eea94b2f49aeb4108106642aeb86/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/1987ca62d99a470f891d5803c8df542d/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/e99cbaa23af042e0ba8dc20b1474d68a/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/20ba31b9febb49c2a959cfd57d234b29/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/287a142f406941019baadef608bb86f4/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/24f1d8a729704c6e9cc1de709f1422f6/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/59dcf1a6916b4d1d806a6fcc12e38121/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/9fc9d8a02bcc4da5bf70ad6b10d5bdaf/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/c3fcfcef6d1546d7b8fe61e39a8a6710/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/d216eeeb75b7468585ef319e942df4d9/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/93cca0e434ac48e9be4108eb2e3dd733/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/4143ccf937e7499b805d030b1751ef85/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/ff94199a3c1d47828dc8d5d9c9987678/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/1083e613359a4aa6a9e7cd81ff94ba02/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/06e8172b7df7463e8febfd5d318c839d/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/9a8a1ae46bf846e8a2b5aa9ece56da7d/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/21d2ac8f6aa04170a9dd0e81b8234010/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/0ad28401783d43ec920a2173c6685728/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/3b24540c9e5f4e5abe0e2256422f778e/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/f02651b8480a4fb6a4aa9eaba3c15ec4/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/9a34cfbd8efd4039a51e9c0c2eee3041/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/df58ae88ac1d46558c2dade2df809355/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/22e7d859bfa34a05a3a0faf7f049ba7f/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/06185cf1cbc54131a178b15d21f9ed58/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/b13ece7471f64a758f85d83c5381c26e/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/b17f54d6c3f549709bdc2f384193941c/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/ebd7b9ef7cfd42c484f2c5af7cdb49ea/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/23763eec87134969b0e2c7ca0ce39b19/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/3dc869b1a98d479caff4de8e2d2c3e13/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/0c5ea884a99e4703996540133d2fb36d/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/0a6bf0cb755d46a39b685bd25776177e/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/26e30a0ae0454fc79ebabe0316916fe9/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/7b26582e075846a095f5dad4de5daced/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/3142630d0ae34bb184219553c5fa3c80/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/ea9bef7238e44e99a445d495b37aa805/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/bdfde90f660e4f8db4ccc1a96025d7a2/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/567eceb840e74b6185a9ad16f9288efb/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/ca7b65a00a174f639061b4a01357d8db/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/69cb67b40ee94b7286b367dc0bedc714/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/c845d1b5517b4034909b2aa5b4aa93e7/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/26f84f33532948e6add58f80e5f6db01/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/689aed01e1c5425db9166950ce94db3e/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/fb00b5463dcd43f0ad6b621ce6ec829a/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/98964f36364345bda33a7f1d4ef172f2/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/1b9aabc65d39450ab625cae678b6d210/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/834b1f6fdcf844219ee95322e831cf6c/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/89297c44234948f6b97f959822d7c2d8/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/644/03ed8f86c38b40d0810a7b4cca04fac8/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="eth_tissue_big_rerun", base="config/final_pixel.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_rep_lr_high",
                            "tissue_rep_lr_high",
                            "tissue_rep_lr_high",
                            "tissue_rep_lr_high",
                            "tissue_rep_lr_normal",
                            "tissue_rep_lr_normal",
                            "tissue_rep_lr_normal",
                            "tissue_rep_lr_normal",
                            "tissue_rep_base",
                            "tissue_rep_base",
                            "tissue_rep_base",
                            "tissue_rep_base",
                            "tissue_rep_keep_10_good_kappa",
                            "tissue_rep_keep_10_good_kappa",
                            "tissue_rep_keep_10_good_kappa",
                            "tissue_rep_keep_10_good_kappa",
                            "tissue_rep_keep_10_best_meaniou",
                            "tissue_rep_keep_10_best_meaniou",
                            "tissue_rep_keep_10_best_meaniou",
                            "tissue_rep_keep_10_best_meaniou",
                            "tissue_rep_keep_30_good_kappa",
                            "tissue_rep_keep_30_good_kappa",
                            "tissue_rep_keep_30_good_kappa",
                            "tissue_rep_keep_30_good_kappa",
                            "tissue_rep_keep_30_best_meaniou",
                            "tissue_rep_keep_30_best_meaniou",
                            "tissue_rep_keep_30_best_meaniou",
                            "tissue_rep_keep_30_best_meaniou",
                            "tissue_rep_weighted_best_meaniou",
                            "tissue_rep_weighted_best_meaniou",
                            "tissue_rep_weighted_best_meaniou",
                            "tissue_rep_weighted_best_meaniou",
                            "tissue_rep_weighted_good_kappa",
                            "tissue_rep_weighted_good_kappa",
                            "tissue_rep_weighted_good_kappa",
                            "tissue_rep_weighted_good_kappa",
                            "tissue_rep_fold_good_meaniou",
                            "tissue_rep_fold_good_meaniou",
                            "tissue_rep_fold_best_meaniou",
                            "tissue_rep_fold_good_kappa",
                            "tissue_rep_fold_good_kappa",
                            "tissue_rep_fold_good_meaniou",
                            "tissue_rep_fold_good_meaniou",
                            "tissue_rep_fold_good_kappa",
                            "tissue_rep_fold_good_kappa",
                            "tissue_rep_fold_best_meaniou",
                            "tissue_rep_fold_best_meaniou",
                            "tissue_rep_fold_best_meaniou",
                            "tissue_rep_fold_best_kappa",
                            "tissue_rep_fold_best_kappa",
                            "tissue_rep_fold_best_kappa",
                            "tissue_rep_fold_best_kappa",
                            "tissue_rep_best_meaniou",
                            "tissue_rep_best_meaniou",
                            "tissue_rep_good_kappa",
                            "tissue_rep_good_kappa",
                            "tissue_rep_good_kappa",
                            "tissue_rep_good_meaniou",
                            "tissue_rep_good_meaniou",
                            "tissue_rep_good_meaniou",
                            "tissue_rep_best_kappa",
                            "tissue_rep_best_kappa",
                            "tissue_rep_best_kappa",
                            "tissue_rep_best_meaniou",
                            "tissue_rep_best_meaniou",
                            "tissue_rep_best_meaniou",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/645/a8bf7412ace54e9385384864ca9e3c83/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/ecc8aadc87934e9c88c206a232dcae8b/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/66bc564f9cd14ce3a1788f1dbf03eacc/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/89139d61e31849f9859f0846640f5748/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/5339c885ed5244d68efb89dc0882bb28/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/4a8d021f72d84695ac67a7895bd32992/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/89e0b32c52ba44ec93084fbb20b634ca/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/e8725f643d784f6688c6e1a503d2099d/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/29c2922008ce4a5680823e6521fb243e/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/39d32a7ccbfb4c9fab8d0eca7ba35889/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/940b1e83e0dc43e2bac552535d9b697f/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/927dcfe79495455887da10b6efb4349c/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/287f23f65f4c46fca662cd5ca7a00a69/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/b8d9128ed99847b49a2e2258f91cb654/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/8f997777a7e54dbbb9f8fb7e86642be9/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/40a74801e469402ea9f680d391380a37/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/e27988e8fe3d4f2ba3a95703ba868378/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/95e6996eecd44be1b336011045503965/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/523264130d6848e7af732ffb1995b964/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/bf66a83f1aba4a73a6a707ff4eecc0a5/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/7cbbf0af7c104860b774f0ae7385d0e8/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/c6de3871564a4f25ba2e0bbb72ace862/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/55177a6dab7e49c0a89574ebb373898f/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/d1828189373f46ddad4586b38c4b99bf/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/adf1cbf97562449eb393b6a4d1811f60/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/56f273571fce46abbfb561653b501274/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/59b703d10ac04f78963713302ac52986/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/7a89c0b464f64cf29beae73b843585d0/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/5058ac68270746c2820acc53f5456af2/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/7ce119865b674858b3e16585588ae44b/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/bb0f9731154f4d1bbdeae425e11a01a6/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/c1c30b0a7a7a4971b8c1abd2cab3d572/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/309da2c062f146df88237fbf70e82e33/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/4a83cdb2ef82460ea3dcb9faafcf6145/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/aaaf056203564911b033add0ff774b0c/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/4c7623dc89e54d7a9f38c2fd4f65c167/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/1e57e71dacd64948b046e3e7fc398500/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/ad59b0452c6e4abc8f85140ab4eed48d/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/4e074ab0729c4057b0bed272c48f35d2/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/2c38a182c7eb4a989380c8fab1856add/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/4216777ae1414e908ec95528d38b2194/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/679ad2c059214de68c60c2338aabca9c/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/8a01c944c8eb4a1293a852b904e72024/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/8e275351282b468c8b1881bcf75421d0/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/8dc8e7989a5e49e29eeeb793828e1304/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/5347027da65d47c8b2c48b5de137cdc1/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/5248209becce410ea0302cf7616dd3ca/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/8df776290d47442884a4f3a8066fc80e/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/cc3666790f4944b1b0bcd2427f9db855/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/4a08b123e6ad4719950f2e65261f2903/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/e9d9de43778a4f61ad4eed061844422d/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/eae73496bfcf4877933187d7222d83d4/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/4021ebae8da042e3802cbfab5bee7a24/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/6cf34a13418e45ee87d6074f8dcbdb46/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/a0f9dd09b47f4a0caa5ac5801a98a480/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/46d614ab20744e3f97f9f68be7d89a99/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/819a33e7136440cb934ba406cc4cd318/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/22ea6ccf4853411bbaba3de668a65cff/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/7f1f177907144e0d891633f553fe5927/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/3a3523d8ef8e4227acffc32618b90f2f/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/525aca9ccf3c4cd69b7eee34ad636319/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/0a5d9ae21d5b4363a49edb7853d4c272/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/4f55632690cc411794f6c1c2d6d1bbbf/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/89a0e82b66ea4dec8342bd9faea12266/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/c0911e38205b4f3fa43d06075ac89c68/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/177db02db93c4016b03a15ce68e55f7f/artifacts/best.valid.node.NodeClassificationF1Score",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="eth_semi_big_rerun", base="config/default_semi.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_v12_ps_combine_13x",
                            "semi_v12_ps_combine_13x",
                            "semi_v12_ps_combine_13x",
                            "semi_v12_ps_combine_13x",
                            "semi_v12_ps_combine_13x",
                            "semi_v12_ps_combine_13x",
                            "semi_v10_ps_med_13x_best_image",
                            "semi_v10_ps_med_13x_best_image",
                            "semi_v10_ps_med_13x_best_image",
                            "semi_v10_ps_med_13x_best_image",
                            "semi_v10_ps_med_13x_best_image",
                            "semi_v11_ps_med_13x_best_pixel",
                            "semi_v11_ps_med_13x_best_pixel",
                            "semi_v11_ps_med_13x_best_pixel",
                            "semi_v11_ps_med_13x_best_pixel",
                            "semi_v11_ps_med_13x_best_pixel",
                            "semi_v10_med_13x_best_image",
                            "semi_v10_med_13x_best_image",
                            "semi_v10_med_13x_best_image",
                            "semi_v10_med_13x_best_image",
                            "semi_v10_med_13x_best_image",
                            "semi_v12_combine_13x",
                            "semi_v12_combine_13x",
                            "semi_v12_combine_13x",
                            "semi_v12_combine_13x",
                            "semi_v12_combine_13x",
                            "semi_v12_combine_13x",
                            "semi_v11_med_13x_best_pixel",
                            "semi_v11_med_13x_best_pixel",
                            "semi_v11_med_13x_best_pixel",
                            "semi_v11_med_13x_best_pixel",
                            "semi_v11_med_13x_best_pixel",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/646/81529c88f3d14deeb28d5228964cdd14/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/b86a3a7516e74130b108a7a2f2762216/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/9e280798a0b0428c98ef0899808b56a4/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/28be542798db4daeb84bcea4fffea4b4/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/d4997ee09f2a469fadaed2fb0182f3cb/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/6c3eb06f631942299b27a4d144ae44dd/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/d7c69ac9e18e461db75f77588d6718ca/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/40487bcef20b482ca1a28bc595fc137c/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/5e9ef5d17acc49c8ad9cd1053b5d64ce/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/05a6b3e2cb4741cb8a77b16a4596b8ad/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/154837d3811c42fda9c66f06ac0f4e43/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/6552799b5f07473da4f54f54d5621427/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/ea39a402a5c5455a9d913b0a3d83dfe3/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/b1f46c0ae1c047cfacfb06d487746034/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/018ce6a161dd44358e75514325a721b9/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/ad50f7d3bd4b42abb027badc4fe31c90/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/deda62170e024d8f9a5f4fbac4d4ead0/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/63d875039e6346ba8d04166a5dcf94b5/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/d482b773a70c4186bb453cb61b384f62/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/40edc5a3eccb4d939ca954cc047b4f66/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/393c3162c18f42c489807a163984e84b/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/13279f315b8d4320b01b055d9748a1d1/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/58b5a54b802d4a57920bde5147e5b93a/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/70acadc92bd241d595c8b0a3e11a2a0f/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/75833660db2646c6a7c298ad36fc3945/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/6137b97464c94286b0fec6cdd27988f7/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/3254ea6f65fc41b5b1aafc22263622a0/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/6cbb0c0b23af4a4ab8c46bc64cbaca21/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/0dc9ff770388433fa8e1342ce24a31e9/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/bf23db0a3cc44ba9b8b379a397e1c165/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/03e3f95d91c44d69aec4960075cfa172/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/646/06600c7e161e47638b2fdb41b6659bd0/artifacts/best.valid.node.NodeClassificationF1Score",
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

    # Pixel Final Fold
    folds = [
        ParameterList(
            ["train", "data", "training_slides"],
            [[111, 199, 204], [76, 111, 199], [204, 76, 111], [199, 204, 76]],
        ),
        ParameterList(
            ["train", "data", "validation_slides"], [[76], [204], [199], [111]]
        ),
    ]
    StronglySupervisedGraphClassificationExperiment(
        name="rep_fold_best_meaniou", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x",
            )
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="rep_fold_good_meaniou", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_very_high_10x",
            )
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="rep_fold_best_kappa", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_low_20x",
            )
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="rep_fold_good_kappa", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_no_13x",
            )
        ],
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_fold_ps_best_meaniou", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v10_mobilenet_med_30x_no_overlap",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
        ],
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_fold_ps_good_meaniou", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_no_10x",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
        ],
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_fold_ps_best_kappa", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_low_13x",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
        ],
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_fold_ps_good_kappa", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_no_10x",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
        ],
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_fold_best_meaniou", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v10_mobilenet_med_30x_no_overlap",
            )
        ],
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_fold_good_meaniou", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_no_10x",
            )
        ],
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_fold_best_kappa", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_low_13x",
            )
        ],
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_fold_good_kappa", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_no_10x",
            )
        ],
        sequential=[folds],
    )

    # Weighted loss
    StronglySupervisedGraphClassificationExperiment(
        name="rep_weighted_best_meaniou", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x",
            ),
            Parameter(["train", "params", "use_weighted_loss"], True),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="rep_weighted_good_meaniou", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_very_high_10x",
            ),
            Parameter(["train", "params", "use_weighted_loss"], True),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="rep_weighted_best_kappa", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_low_20x",
            ),
            Parameter(["train", "params", "use_weighted_loss"], True),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="rep_weighted_good_kappa", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_no_13x",
            ),
            Parameter(["train", "params", "use_weighted_loss"], True),
        ],
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_weighted_ps_best_meaniou", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v10_mobilenet_med_30x_no_overlap",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
            Parameter(["train", "params", "use_weighted_loss"], True),
        ],
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_weighted_ps_good_meaniou", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_no_10x",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
            Parameter(["train", "params", "use_weighted_loss"], True),
        ],
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_weighted_ps_best_kappa", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_low_13x",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
            Parameter(["train", "params", "use_weighted_loss"], True),
        ],
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_weighted_ps_good_kappa", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_no_10x",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
            Parameter(["train", "params", "use_weighted_loss"], True),
        ],
        sequential=[folds],
    )

    # Node stochastic
    StronglySupervisedGraphClassificationExperiment(
        name="rep_keep_30_best_meaniou", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x",
            ),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 30
            ),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="rep_keep_30_good_kappa", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_no_13x",
            ),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 30
            ),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="rep_keep_10_best_meaniou", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x",
            ),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 10
            ),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="rep_keep_10_good_kappa", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_no_13x",
            ),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 10
            ),
        ],
        sequential=[folds],
    )

    # New base
    StronglySupervisedGraphClassificationExperiment(
        name="rep_base", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x",
            ),
            Parameter(["train", "params", "focused_metric"], "fF1Score"),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="rep_lr_normal", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x",
            ),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.0000001,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.00003,
            ),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "focused_metric"], "fF1Score"),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="rep_lr_high", base="config/final_pixel.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x",
            ),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 5,
                        "min_lr": 0.000001,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.001,
            ),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "focused_metric"], "fF1Score"),
        ],
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_lr_normal", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_low_13x",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.0000001,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.00003,
            ),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "focused_metric"], "fF1Score"),
        ],
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_lr_high", base="config/final_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v12_mobilenet_low_13x",
            ),
            Parameter(["train", "data", "image_labels_mode"], "p+s"),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 5,
                        "min_lr": 0.000001,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.001,
            ),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "focused_metric"], "fF1Score"),
        ],
        sequential=[folds],
    )

    # FINAL CNN (ETH Parameters)
    PretrainingExperiment(
        name="rep_final",
        base="config/final_cnn.yml",
        queue="prod.long",
        path=PATH,
    ).generate(
        sequential=[folds],
    )
    PretrainingExperiment(
        name="rep_final2",
        base="config/final_cnn4.yml",
        queue="prod.long",
        path=PATH,
    ).generate(
        sequential=[folds],
    )

    # Final semi-supervised
    SemiSupervisedGraphClassificationExperiment(
        name="rep_semi_0.7", path=PATH
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
            Parameter(["train", "params", "loss", "node_weight"], 0.7),
        ],
        sequential=[folds],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="rep_semi_0.5", path=PATH
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
            Parameter(["train", "params", "loss", "node_weight"], 0.5),
        ],
        sequential=[folds],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="rep_semi_0.9", path=PATH
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
            Parameter(["train", "params", "loss", "node_weight"], 0.9),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="rep_semi_node_compare", base="config/default_strong.yml", path=PATH
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
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_semi_graph_compare", base="config/default_weak2.yml", path=PATH
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
        ],
        sequential=[folds],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="rep_semi_0.3", path=PATH
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
            Parameter(["train", "params", "loss", "node_weight"], 0.3),
        ],
        sequential=[folds],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="rep_semi_0.1", path=PATH
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
            Parameter(["train", "params", "loss", "node_weight"], 0.1),
        ],
        sequential=[folds],
    )

    CNNTestingExperiment(name="rerun_final", path=PATH).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "rep_final",
                            "rep_final",
                            "rep_final",
                            "rep_final",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        f"s3://mlflow/633/{run}/artifacts/best.valid.MeanMultiLabelF1Score"
                        for run in [
                            "ca9c816ef60d4ad4bd22bd0ba99313f2",
                            "5f5e761e41634fdd8d0ad103667c6e99",
                            "065eeb18f01846c483d6462bf3ae4836",
                            "7b70ffbe772f49b5b38dcf63eda688a5",
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
    CNNTestingExperiment(name="rerun_final2", path=PATH).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "rep_final2",
                            "rep_final2",
                            "rep_final2",
                            "rep_final2",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        f"s3://mlflow/633/{run}/artifacts/best.valid.MeanMultiLabelF1Score"
                        for run in [
                            "42f2d4fd7f6a43348b4bf472391abe78",
                            "6e72f780d75f4f4b9a710287f19cf319",
                            "0833705e25a74eb8b4bb26cc912c03a4",
                            "de65c0a390e84266a769a18173f20d9a",
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
        name="eth_rerun_semi_node", base="config/default_strong.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_rep_node_0.9",
                            "semi_rep_node_0.9",
                            "semi_rep_node_0.9",
                            "semi_rep_node_0.9",
                            "semi_rep_node_0.5",
                            "semi_rep_node_0.5",
                            "semi_rep_node_0.5",
                            "semi_rep_node_0.5",
                            "semi_rep_node_0.7",
                            "semi_rep_node_0.7",
                            "semi_rep_node_0.7",
                            "semi_rep_node_0.7",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/646/3d498ccc11384fd58248d763be09fcd0/artifacts/best.valid.node.segmentation.MeanF1Score",
                        "s3://mlflow/646/24af2c15568341f2914366e38d893376/artifacts/best.valid.node.segmentation.MeanF1Score",
                        "s3://mlflow/646/9c47ee6c52c44d6ea65661b1cbf3c615/artifacts/best.valid.node.segmentation.MeanF1Score",
                        "s3://mlflow/646/f5863b829ce64da8ab98e82c93650dca/artifacts/best.valid.node.segmentation.MeanF1Score",
                        "s3://mlflow/646/0a8e5afd9b74432eb8d687b9d2416f5e/artifacts/best.valid.node.segmentation.MeanF1Score",
                        "s3://mlflow/646/7cf186e3071447f7b20fa891ef2124ee/artifacts/best.valid.node.segmentation.MeanF1Score",
                        "s3://mlflow/646/87400401128f4fc1b5f16535eb9d4f37/artifacts/best.valid.node.segmentation.MeanF1Score",
                        "s3://mlflow/646/880fa612b4484e179e3d36bc28a08bf8/artifacts/best.valid.node.segmentation.MeanF1Score",
                        "s3://mlflow/646/80cf3f27bfdc437d8665c5980031bdb1/artifacts/best.valid.node.segmentation.MeanF1Score",
                        "s3://mlflow/646/afe19dfe448b4f74985194445aedc226/artifacts/best.valid.node.segmentation.MeanF1Score",
                        "s3://mlflow/646/09d8e0f469df4186aadd5ca25967c2d1/artifacts/best.valid.node.segmentation.MeanF1Score",
                        "s3://mlflow/646/b27d6bbf4ed64fc29ce5d97708ccb173/artifacts/best.valid.node.segmentation.MeanF1Score",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="eth_rerun_semi_graph", base="config/default_weak.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_rep_graph_0.9",
                            "semi_rep_graph_0.9",
                            "semi_rep_graph_0.9",
                            "semi_rep_graph_0.9",
                            "semi_rep_graph_0.5",
                            "semi_rep_graph_0.5",
                            "semi_rep_graph_0.5",
                            "semi_rep_graph_0.5",
                            "semi_rep_graph_0.7",
                            "semi_rep_graph_0.7",
                            "semi_rep_graph_0.7",
                            "semi_rep_graph_0.7",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/646/3d498ccc11384fd58248d763be09fcd0/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/646/24af2c15568341f2914366e38d893376/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/646/9c47ee6c52c44d6ea65661b1cbf3c615/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/646/f5863b829ce64da8ab98e82c93650dca/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/646/0a8e5afd9b74432eb8d687b9d2416f5e/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/646/7cf186e3071447f7b20fa891ef2124ee/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/646/87400401128f4fc1b5f16535eb9d4f37/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/646/880fa612b4484e179e3d36bc28a08bf8/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/646/80cf3f27bfdc437d8665c5980031bdb1/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/646/afe19dfe448b4f74985194445aedc226/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/646/09d8e0f469df4186aadd5ca25967c2d1/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/646/b27d6bbf4ed64fc29ce5d97708ccb173/artifacts/best.valid.graph.segmentation.MeanF1Score",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="eth_rerun_semi_new_graph", base="config/default_weak.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_rep_graph_0.1",
                            "semi_rep_graph_0.1",
                            "semi_rep_graph_0.1",
                            "semi_rep_graph_0.1",
                            "semi_rep_graph_0.3",
                            "semi_rep_graph_0.3",
                            "semi_rep_graph_0.3",
                            "semi_rep_graph_0.3",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/646/c7567ec065b34d3f93cf800c345b82e2/artifacts/best.valid.graph.segmentation.fF1Score",
                        "s3://mlflow/646/e4bb4540ea7f413e916c875c96a39d63/artifacts/best.valid.graph.segmentation.fF1Score",
                        "s3://mlflow/646/1372b6afbf2c4d18bac885225745be28/artifacts/best.valid.graph.segmentation.fF1Score",
                        "s3://mlflow/646/4fb70da099f04682a45d82a7539c2964/artifacts/best.valid.graph.segmentation.fF1Score",
                        "s3://mlflow/646/518e543945374d9499b60874cf084325/artifacts/best.valid.graph.segmentation.fF1Score",
                        "s3://mlflow/646/865a6064e2cd45118a2ac5a679dafeb4/artifacts/best.valid.graph.segmentation.fF1Score",
                        "s3://mlflow/646/42ee5440d57c45d9a43b1e42bbac47b6/artifacts/best.valid.graph.segmentation.fF1Score",
                        "s3://mlflow/646/9b9320e7017149df84e5151ccf2417a1/artifacts/best.valid.graph.segmentation.fF1Score",
                    ],
                ),
            ]
        ],
    )

    # Partial Annotations
    CPUPreprocessingExperiment(
        name="v11_standard_med_13x_partial", base="config/new_preprocess.yml", path=PATH
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
            Parameter(
                ["superpixel", "params", "nr_superpixels"],
                400,
            ),
            Parameter(
                ["superpixel", "params", "compactness"],
                30,
            ),
            Parameter(
                ["feature_extraction", "params", "downsample_factor"],
                3,
            ),
        ],
        sequential=[
            [
                ParameterList(
                    ["graph_builders", "params", "partial_annotation"], [50, 25]
                ),
                ParameterList(["params", "partial_annotation"], [50, 25]),
                ParameterList(
                    ["params", "link_directory"],
                    [f"v11_mobilenet_med_13x_partial_{s}" for s in [50, 25]],
                ),
            ]
        ],
    )
    CPUPreprocessingExperiment(
        name="v11_standard_med_13x_partial_new",
        base="config/new_preprocess.yml",
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
                32,
            ),
            Parameter(["superpixel", "params", "threshold"], 0.02),
            Parameter(
                ["superpixel", "params", "nr_superpixels"],
                400,
            ),
            Parameter(
                ["superpixel", "params", "compactness"],
                30,
            ),
            Parameter(
                ["feature_extraction", "params", "downsample_factor"],
                3,
            ),
        ],
        sequential=[
            [
                ParameterList(
                    ["graph_builders", "params", "partial_annotation"], [10, 5]
                ),
                ParameterList(["params", "partial_annotation"], [10, 5]),
                ParameterList(
                    ["params", "link_directory"],
                    [f"v11_mobilenet_med_13x_partial_{s}" for s in [10, 5]],
                ),
            ]
        ],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="eth_partial_50",
        base="config/paper_eth_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_50",
            ),
            Parameter(["train", "data", "partial_annotation"], 50),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="eth_partial_25",
        base="config/paper_eth_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_25",
            ),
            Parameter(["train", "data", "partial_annotation"], 25),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="eth_partial_10",
        base="config/paper_eth_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_10",
            ),
            Parameter(["train", "data", "partial_annotation"], 10),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="eth_partial_5",
        base="config/paper_eth_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_5",
            ),
            Parameter(["train", "data", "partial_annotation"], 5),
        ],
        sequential=[folds],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="eth_partial_50", base="config/paper_eth_semi.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_50",
            ),
            Parameter(["train", "data", "partial_annotation"], 50),
        ],
        sequential=[folds],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="eth_partial_25", base="config/paper_eth_semi.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_25",
            ),
            Parameter(["train", "data", "partial_annotation"], 25),
        ],
        sequential=[folds],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="eth_partial_10", base="config/paper_eth_semi.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_10",
            ),
            Parameter(["train", "data", "partial_annotation"], 10),
        ],
        sequential=[folds],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="eth_partial_5", base="config/paper_eth_semi.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_5",
            ),
            Parameter(["train", "data", "partial_annotation"], 5),
        ],
        sequential=[folds],
    )

    GNNTestingExperiment(
        name="eth_rerun_new_partial_tissue",
        base="config/paper_eth_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_eth_partial_5",
                            "tissue_eth_partial_5",
                            "tissue_eth_partial_5",
                            "tissue_eth_partial_5",
                            "tissue_eth_partial_10",
                            "tissue_eth_partial_10",
                            "tissue_eth_partial_10",
                            "tissue_eth_partial_10",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/645/4bf6a4679f014ed6b575303d6596641b/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/adec59159dd4483688af542d27dc961a/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/64309db47967415caaad16d08410bd43/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/14728435a9ba4eb1b6151d4ce4dd0aed/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/f8840786a1834569a058224ff7f59e31/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/30eff8f8135c4407bdc9fc6dcc5481da/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/ab91980014f145eaae115f02e5eb47e6/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/a66a07a6a6b24a86873c4f2d01a4a2b8/artifacts/best.valid.node.segmentation.fF1Score",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="eth_rerun_new_partial_semi", base="config/paper_eth_semi.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_eth_partial_5",
                            "semi_eth_partial_5",
                            "semi_eth_partial_5",
                            "semi_eth_partial_5",
                            "semi_eth_partial_10",
                            "semi_eth_partial_10",
                            "semi_eth_partial_10",
                            "semi_eth_partial_10",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/646/7dfde04f6f734a01a721dd3e1fd9f001/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/9fc37069990a44c0b3ef52a0d35586c8/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/edbecac3a4c34969bb5d064ac9428cc0/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/3a2b9fc4c83646cd9113d49d4ce0282d/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/c126dbd83c4f43719cc836679b912974/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/76eee0bd41da47d89d268468d5483ee1/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/a2828465f3ef4e1a82201b965f9b510f/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/fbb1cf0401774debac8e0274a1d973e6/artifacts/best.valid.node.segmentation.fF1Score",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="eth_rerun_new_partial_semi_node",
        base="config/paper_eth_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_eth_partial_node_5",
                            "semi_eth_partial_node_5",
                            "semi_eth_partial_node_5",
                            "semi_eth_partial_node_5",
                            "semi_eth_partial_node_10",
                            "semi_eth_partial_node_10",
                            "semi_eth_partial_node_10",
                            "semi_eth_partial_node_10",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/646/7dfde04f6f734a01a721dd3e1fd9f001/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/9fc37069990a44c0b3ef52a0d35586c8/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/edbecac3a4c34969bb5d064ac9428cc0/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/3a2b9fc4c83646cd9113d49d4ce0282d/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/c126dbd83c4f43719cc836679b912974/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/76eee0bd41da47d89d268468d5483ee1/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/a2828465f3ef4e1a82201b965f9b510f/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/fbb1cf0401774debac8e0274a1d973e6/artifacts/best.valid.node.segmentation.fF1Score",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="eth_rerun_new_partial_tissue_high",
        base="config/paper_eth_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_eth_partial_5",
                            "tissue_eth_partial_5",
                            "tissue_eth_partial_5",
                            "tissue_eth_partial_5",
                            "tissue_eth_partial_10",
                            "tissue_eth_partial_10",
                            "tissue_eth_partial_10",
                            "tissue_eth_partial_10",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/645/4bf6a4679f014ed6b575303d6596641b/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/adec59159dd4483688af542d27dc961a/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/64309db47967415caaad16d08410bd43/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/14728435a9ba4eb1b6151d4ce4dd0aed/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/f8840786a1834569a058224ff7f59e31/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/30eff8f8135c4407bdc9fc6dcc5481da/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/ab91980014f145eaae115f02e5eb47e6/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/a66a07a6a6b24a86873c4f2d01a4a2b8/artifacts/best.valid.node.segmentation.fF1Score",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="eth_rerun_new_partial_semi_high",
        base="config/paper_eth_semi.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x + "_graph"}
                        for x in [
                            "semi_eth_partial_25",
                            "semi_eth_partial_25",
                            "semi_eth_partial_25",
                            "semi_eth_partial_50",
                            "semi_eth_partial_25",
                            "semi_eth_partial_50",
                            "semi_eth_partial_50",
                            "semi_eth_partial_50",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/646/9f5b61f1738d4e3cbb7be3970edb608a/artifacts/best.valid.graph.segmentation.fF1Score",
                        "s3://mlflow/646/825e329a92544921a3a48f0a108ed74c/artifacts/best.valid.graph.segmentation.fF1Score",
                        "s3://mlflow/646/fbe9ee7cbecf4e788f574a1a07e266d4/artifacts/best.valid.graph.segmentation.fF1Score",
                        "s3://mlflow/646/20a6c323b81d4268a4012791cfa60840/artifacts/best.valid.graph.segmentation.fF1Score",
                        "s3://mlflow/646/3242af9de85446df8932a2e1d2cfbf50/artifacts/best.valid.graph.segmentation.fF1Score",
                        "s3://mlflow/646/ac268b7a01694a99bcb00b96cd2da572/artifacts/best.valid.graph.segmentation.fF1Score",
                        "s3://mlflow/646/7d530190c66e4cbcac3a6b85cfe3f993/artifacts/best.valid.graph.segmentation.fF1Score",
                        "s3://mlflow/646/94f15c24601b4ce286e73711a9da53d0/artifacts/best.valid.graph.segmentation.fF1Score",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="eth_rerun_new_partial_semi_high_node",
        base="config/paper_eth_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x+ "_node"}
                        for x in [
                            "semi_eth_partial_25",
                            "semi_eth_partial_25",
                            "semi_eth_partial_25",
                            "semi_eth_partial_50",
                            "semi_eth_partial_25",
                            "semi_eth_partial_50",
                            "semi_eth_partial_50",
                            "semi_eth_partial_50",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/646/9f5b61f1738d4e3cbb7be3970edb608a/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/825e329a92544921a3a48f0a108ed74c/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/fbe9ee7cbecf4e788f574a1a07e266d4/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/20a6c323b81d4268a4012791cfa60840/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/3242af9de85446df8932a2e1d2cfbf50/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/ac268b7a01694a99bcb00b96cd2da572/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/7d530190c66e4cbcac3a6b85cfe3f993/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/646/94f15c24601b4ce286e73711a9da53d0/artifacts/best.valid.node.segmentation.fF1Score",
                    ],
                ),
            ]
        ],
    )

    StronglySupervisedGraphClassificationExperiment(
        name="rep_final_strong", base="config/paper_eth_strong.yml", path=PATH
    ).generate(
        sequential=[folds],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="rep_final_weak", base="config/paper_eth_weak.yml", path=PATH
    ).generate(
        sequential=[folds],
    )
