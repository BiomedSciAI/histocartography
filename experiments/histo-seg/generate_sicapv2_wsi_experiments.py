import argparse
from pathlib import Path

import numpy as np

from experiment import (
    CPUPreprocessingExperiment,
    GNNTestingExperiment,
    GPUPreprocessingExperiment,
    Parameter,
    ParameterList,
    SemiSupervisedGraphClassificationExperiment,
    StronglySupervisedGraphClassificationExperiment,
    WeaklySupervisedGraphClassificationExperiment,
    PATH,
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
        path=PATH,
    ).generate()
    CPUPreprocessingExperiment(
        name="sicapv2_wsi",
        base="config/preprocessing_sicap_wsi.yml",
        cores=4,
        path=PATH,
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
    StronglySupervisedGraphClassificationExperiment(
        name="best_strong_node_aug", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="best_weak_node_aug", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="best_strong_no_location", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="best_weak_no_location", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="best_weak_more_dropout", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        sequential=[
            ParameterList(
                ["train", "model", "graph_classifier_config", "input_dropout"],
                [0.1, 0.3, 0.5, 0.7],
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="best_weak_anti_overfit", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "model", "graph_classifier_config", "input_dropout"], 0.3
            ),
            Parameter(["train", "model", "gnn_config", "n_layers"], 6),
            Parameter(["train", "model", "gnn_config", "dropout"], 0.7),
        ],
        sequential=[
            [
                ParameterList(
                    ["train", "model", "gnn_config", "hidden_dim"],
                    [16, 32, 64],
                ),
                ParameterList(
                    ["train", "model", "gnn_config", "output_dim"],
                    [16, 32, 64],
                ),
            ]
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="best_weak_optim_anti_overfit",
        base="config/sicapv2_wsi_weak.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "model", "graph_classifier_config", "input_dropout"], 0.3
            ),
            Parameter(["train", "model", "gnn_config", "n_layers"], 6),
            Parameter(["train", "model", "gnn_config", "dropout"], 0.7),
            Parameter(["train", "params", "optimizer", "params", "lr"], 0.00003),
            Parameter(["train", "params", "optimizer", "scheduler"], None),
        ],
        sequential=[
            [
                ParameterList(
                    ["train", "model", "gnn_config", "hidden_dim"],
                    [16, 32, 64],
                ),
                ParameterList(
                    ["train", "model", "gnn_config", "output_dim"],
                    [16, 32, 64],
                ),
            ]
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    # Weighted loss
    StronglySupervisedGraphClassificationExperiment(
        name="best_weighted_strong", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="best_weighted_strong_node_aug",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )

    # Keep nodes
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_keep_200_best_strong", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"],
                200
            )
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_keep_100_best_strong", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"],
                100
            )
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_keep_half_best_strong", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(
                ["train", "params", "loss", "node", "params", "drop_probability"],
                0.5
            )
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_keep_quarter_best_strong", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(
                ["train", "params", "loss", "node", "params", "drop_probability"],
                0.75
            )
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    GNNTestingExperiment(
        name="tissue_failed", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_best_strong",
                            "tissue_best_strong",
                            "tissue_best_strong",
                            "tissue_best_strong",
                            "tissue_best_strong_node_aug",
                            "tissue_best_strong_node_aug",
                            "tissue_best_strong_node_aug",
                            "tissue_best_strong_node_aug",
                            "tissue_best_strong_no_location",
                            "tissue_best_strong_no_location",
                            "tissue_best_strong_no_location",
                            "tissue_best_strong_no_location",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        f"s3://mlflow/650/{run}/artifacts/best.valid.node.segmentation.MeanIoU"
                        for run in [
                            "4b7e18f29d3a49b398213640c8c09237",
                            "bc8558fca31846129bd51d560ba094a3",
                            "fc3f866e4dfa49469c0f730fec970c5a",
                            "fca69870629d47acba7ef31218f78fbe",
                            "26a9a973906c454b8a757c5266e0c0ff",
                            "b3be7493ae4d486e911e9b25154508f8",
                            "61a97fc20e8a48a68b4e6c4a0550f8bf",
                            "6a6fa62c8a37484aa655517c1b8862cd",
                            "bfb3fc22dce3424ba88614d6d7bcecbd",
                            "ae849c966b8743e4bfcd3a332f9656cf",
                            "0678ff999d7b442da03165865adb9add",
                            "c4499129b23144af96ba62f21fa27dd6",
                        ]
                    ],
                ),
            ]
        ],
    )
