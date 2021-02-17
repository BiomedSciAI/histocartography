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
    GPUPreprocessingExperiment(
        name="sicapv2_resnet34",
        base="config/feat_sicap_wsi.yml",
        queue="prod.p9",
        workers=24,
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                [
                    "pipeline",
                    "stages",
                    3,
                    "feature_extraction",
                    "params",
                    "architecture",
                ],
                "resnet34",
            )
        ]
    )
    CPUPreprocessingExperiment(
        name="sicapv2_resnet34_rest",
        base="config/preprocessing_sicap_wsi.yml",
        cores=4,
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["params", "link_directory"],
                "v0_low_4000_resnet34",
            ),
            Parameter(
                [
                    "pipeline",
                    "stages",
                    3,
                    "feature_extraction",
                    "params",
                    "architecture",
                ],
                "resnet34",
            ),
        ]
    )
    GPUPreprocessingExperiment(
        name="sicapv2_pretrained_cnn",
        base="config/feat_sicap_wsi.yml",
        queue="prod.p9",
        workers=24,
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                [
                    "pipeline",
                    "stages",
                    3,
                    "feature_extraction",
                    "params",
                    "architecture",
                ],
                "s3://mlflow/633/734fc44a6db048f5a081c33d0ba07428/artifacts/best.valid.MultiLabelBalancedAccuracy",
            )
        ]
    )
    CPUPreprocessingExperiment(
        name="sicapv2_pretrained_rest",
        base="config/preprocessing_sicap_wsi.yml",
        cores=4,
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["params", "link_directory"],
                "v0_low_4000_pretrained",
            ),
            Parameter(
                [
                    "pipeline",
                    "stages",
                    3,
                    "feature_extraction",
                    "params",
                    "architecture",
                ],
                "s3://mlflow/633/734fc44a6db048f5a081c33d0ba07428/artifacts/best.valid.MultiLabelBalancedAccuracy",
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
        name="sicap_best_weighted_strong",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_best_weighted_strong_node_aug",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_best_weighted_strong_keep",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 200
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_best_weighted_strong_node_aug_keep",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 200
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_best_lin_weights",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], False),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_best_lin_weights_keep",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], False),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 200
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_best_lin_oversample_keep",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], False),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 200
            ),
            Parameter(["train", "params", "balanced_sampling"], True),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_best_log_oversample_keep",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], True),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 200
            ),
            Parameter(["train", "params", "balanced_sampling"], True),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )

    # Keep nodes
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_keep_200_best_strong",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 200
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_keep_100_best_strong",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 100
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_keep_half_best_strong",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(
                ["train", "params", "loss", "node", "params", "drop_probability"], 0.5
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_keep_quarter_best_strong",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(
                ["train", "params", "loss", "node", "params", "drop_probability"], 0.75
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    # Patch size
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_best_keep_100_patches_1000",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], True),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 100
            ),
            Parameter(["train", "params", "balanced_sampling"], True),
            Parameter(["train", "data", "patch_size"], 1000),
            Parameter(["train", "params", "nr_epochs"], 20000),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_best_keep_200_patches_2000",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], True),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 200
            ),
            Parameter(["train", "params", "balanced_sampling"], True),
            Parameter(["train", "data", "patch_size"], 2000),
            Parameter(["train", "params", "nr_epochs"], 15000),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_best_keep_300_patches_3000",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], True),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 300
            ),
            Parameter(["train", "params", "balanced_sampling"], True),
            Parameter(["train", "data", "patch_size"], 3000),
            Parameter(["train", "params", "nr_epochs"], 10000),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_best_only_patches_1000",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], True),
            Parameter(["train", "params", "balanced_sampling"], True),
            Parameter(["train", "data", "patch_size"], 1000),
            Parameter(["train", "params", "nr_epochs"], 20000),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_best_only_patches_2000",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], True),
            Parameter(["train", "params", "balanced_sampling"], True),
            Parameter(["train", "data", "patch_size"], 2000),
            Parameter(["train", "params", "nr_epochs"], 15000),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_best_only_patches_3000",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], True),
            Parameter(["train", "params", "balanced_sampling"], True),
            Parameter(["train", "data", "patch_size"], 3000),
            Parameter(["train", "params", "nr_epochs"], 10000),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
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
    GNNTestingExperiment(
        name="sicap_rerun_failed", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_sicap_best_weighted_strong_keep",
                            "tissue_sicap_best_weighted_strong_node_aug_keep",
                            "tissue_sicap_best_weighted_strong_node_aug_keep",
                            "tissue_sicap_best_weighted_strong_node_aug_keep",
                            "tissue_sicap_best_weighted_strong_node_aug_keep",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        f"s3://mlflow/650/{run}/artifacts/best.valid.node.segmentation.MeanIoU"
                        for run in [
                            "30ecbfa18b364f1da861b72a2c130788",
                            "e59352133a2d4c41abe7f8a241e81b34",
                            "f582548a20ef43e089e022f45780c0f4",
                            "c2cadd9218b84e389ebf3b138966edc1",
                            "ab0b5264549044a79a5ba98d531cd7ca",
                        ]
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="sicap_new_failed", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_sicap_best_lin_weights_keep",
                            "tissue_sicap_best_lin_weights_keep",
                            "tissue_sicap_best_lin_weights_keep",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        f"s3://mlflow/650/{run}/artifacts/best.valid.node.segmentation.fF1Score"
                        for run in [
                            "a05ccc1498f2473cb710e496018f8ae7",
                            "9609efffbde64ff8b60b69b23c821c07",
                            "d71b071a64b8432b85fb9f2ace63bf1e",
                        ]
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="sicap_big_rerun", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_sicap_best_lin_weights_keep",
                            "tissue_sicap_best_lin_weights_keep",
                            "tissue_sicap_best_lin_weights_keep",
                            "tissue_sicap_best_lin_weights",
                            "tissue_sicap_best_lin_weights",
                            "tissue_sicap_best_lin_weights",
                            "tissue_sicap_best_lin_weights",
                            "tissue_sicap_best_weighted_strong_node_aug_keep",
                            "tissue_sicap_best_weighted_strong_node_aug_keep",
                            "tissue_sicap_best_weighted_strong_node_aug_keep",
                            "tissue_sicap_best_weighted_strong_node_aug_keep",
                            "tissue_sicap_best_weighted_strong_keep",
                            "tissue_sicap_best_weighted_strong_keep",
                            "tissue_sicap_best_weighted_strong_keep",
                            "tissue_sicap_best_weighted_strong_keep",
                            "tissue_sicap_best_weighted_strong_node_aug",
                            "tissue_sicap_best_weighted_strong_node_aug",
                            "tissue_sicap_best_weighted_strong_node_aug",
                            "tissue_sicap_best_weighted_strong_node_aug",
                            "tissue_sicap_best_weighted_strong",
                            "tissue_sicap_best_weighted_strong",
                            "tissue_sicap_best_weighted_strong",
                            "tissue_sicap_best_weighted_strong",
                            "tissue_sicap_keep_quarter_best_strong",
                            "tissue_sicap_keep_quarter_best_strong",
                            "tissue_sicap_keep_quarter_best_strong",
                            "tissue_sicap_keep_quarter_best_strong",
                            "tissue_sicap_keep_half_best_strong",
                            "tissue_sicap_keep_half_best_strong",
                            "tissue_sicap_keep_half_best_strong",
                            "tissue_sicap_keep_half_best_strong",
                            "tissue_sicap_keep_100_best_strong",
                            "tissue_sicap_keep_100_best_strong",
                            "tissue_sicap_keep_200_best_strong",
                            "tissue_sicap_keep_200_best_strong",
                            "tissue_sicap_keep_200_best_strong",
                            "tissue_sicap_keep_200_best_strong",
                            "tissue_best_weighted_strong_node_aug",
                            "tissue_best_weighted_strong_node_aug",
                            "tissue_best_weighted_strong_node_aug",
                            "tissue_best_weighted_strong",
                            "tissue_best_weighted_strong_node_aug",
                            "tissue_best_weighted_strong",
                            "tissue_best_weighted_strong",
                            "tissue_best_weighted_strong",
                            "tissue_best_strong_no_location",
                            "tissue_best_strong_no_location",
                            "tissue_best_strong_no_location",
                            "tissue_best_strong_no_location",
                            "tissue_best_strong_node_aug",
                            "tissue_best_strong_node_aug",
                            "tissue_best_strong_node_aug",
                            "tissue_best_strong_node_aug",
                            "tissue_best_strong",
                            "tissue_best_strong",
                            "tissue_best_strong",
                            "tissue_best_strong",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/650/a05ccc1498f2473cb710e496018f8ae7/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/9609efffbde64ff8b60b69b23c821c07/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/d71b071a64b8432b85fb9f2ace63bf1e/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/21f38c8448f949af8e55f0fac5d5fe76/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/fc0671e44e79467c877bfb21936d3e64/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/b1297baf4a95447c80abd7573a41fbd7/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/62db4c1af64f46bc9416777d785d766c/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/c2cadd9218b84e389ebf3b138966edc1/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/f582548a20ef43e089e022f45780c0f4/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/ab0b5264549044a79a5ba98d531cd7ca/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/e59352133a2d4c41abe7f8a241e81b34/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/30ecbfa18b364f1da861b72a2c130788/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/b3d287ff833d4f188c0114e7f3b09251/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/3a70095b947e40c58918f5581e530c8f/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/225184819f60492aac0a31d1a60324b0/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/abeb9f7cfdf44aa492449b8b088b92f3/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/4d4fe9e128044b63b50d94c53092a485/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/66bd40bdaa7a47409576c971d554ee5a/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/72bf68b1ef6b485494a685a606dc5266/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/79d146447bdb4557a3ae3dc21a7efdc0/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/623660caef46474fb5b73f843fa2a93c/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/47d1f7ffb85b4d4e8c71996bb8068b87/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/e8ded6ad8a1742d6b422df9385db0a47/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/0c008ca6d9d14b55b4cdb20afc727397/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/1456096fd7fc467cbb1fe463eb92ae16/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/275ddb0f19fc4c1ba1c9460f29d906af/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/c48ba4679f2c4390ada010d46a056381/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/062b1a337c6a40e8adc192016f0e5d32/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/0ec62545877f4ffe97018190c1f3cffe/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/668d30aa6666441c8ace693fd769e18d/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/e7e6211332a84e52808f626c1d113d16/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/7f96b35953884004a7f269f8b2f3f916/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/a825e0f6bffc4223bc45712474f7324a/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/5cee1e4410a94d9cb7d111e50a13ff72/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/267df47a09ff48939e0c3ccef47a7903/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/e038d6be894047c2b996d0fed5eab024/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/ec5a8cb5e9df4e298e6c7cd118d9705e/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/9cdbea6d8727467d91bc64f2bc002a12/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/a1739a26f7ca45c0b9a7e116562e7cdf/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/b160a07a8b854f6eb71388476d070ef6/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/8afe0d60977647fdba8e8e7f20972112/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/c3d0450979a84b80a63c01be4dd2ab9f/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/a0ec2b7ab5fa4e17baa15ab15873096c/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/becac678ddae4e5e8c6511d6487811d8/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/6bdea14270b440028c4ee42b991918ed/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/bfb3fc22dce3424ba88614d6d7bcecbd/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/ae849c966b8743e4bfcd3a332f9656cf/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/0678ff999d7b442da03165865adb9add/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/c4499129b23144af96ba62f21fa27dd6/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/26a9a973906c454b8a757c5266e0c0ff/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/b3be7493ae4d486e911e9b25154508f8/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/61a97fc20e8a48a68b4e6c4a0550f8bf/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/6a6fa62c8a37484aa655517c1b8862cd/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/4b7e18f29d3a49b398213640c8c09237/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/bc8558fca31846129bd51d560ba094a3/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/fc3f866e4dfa49469c0f730fec970c5a/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/fca69870629d47acba7ef31218f78fbe/artifacts/best.valid.node.NodeClassificationF1Score",
                    ],
                ),
            ]
        ],
    )

    # ResNet34
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_resnet_weighted_node_aug",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "graph_directory"], "v0_low_4000_resnet34"),
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 512),
            Parameter(["train", "params", "use_weighted_loss"], True),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_resnet_weighted_node_aug_keep",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "graph_directory"], "v0_low_4000_resnet34"),
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 512),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 200
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )

    # Pretrained
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_pretrained_weighted_node_aug",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "graph_directory"], "v0_low_4000_pretrained"),
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_pretrained_weighted_node_aug_keep",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "graph_directory"], "v0_low_4000_pretrained"),
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 200
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )

    # Semi supervised
    SemiSupervisedGraphClassificationExperiment(
        name="sicap_semi_0.5",
        base="config/sicapv2_wsi_semi.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "data", "augmentation_mode"], "node"),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "params", "loss", "node_weight"], [0.5])],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="sicap_semi_0.25",
        base="config/sicapv2_wsi_semi.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "data", "augmentation_mode"], "node"),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "params", "loss", "node_weight"], [0.25])],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="sicap_semi_0.75",
        base="config/sicapv2_wsi_semi.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "data", "augmentation_mode"], "node"),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "params", "loss", "node_weight"], [0.75])],
    )
