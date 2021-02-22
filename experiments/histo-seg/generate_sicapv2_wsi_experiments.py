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
    GPUPreprocessingExperiment(
        name="sicapv2_fsconv_1",
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
                "s3://mlflow/7683e7dbefcd499096fc03ba84353c05/artifacts/model_1.pt",
            )
        ]
    )
    GPUPreprocessingExperiment(
        name="sicapv2_fsconv_2",
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
                "s3://mlflow/7683e7dbefcd499096fc03ba84353c05/artifacts/model_2.pt",
            )
        ]
    )
    GPUPreprocessingExperiment(
        name="sicapv2_fsconv_3",
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
                "s3://mlflow/7683e7dbefcd499096fc03ba84353c05/artifacts/model_3.pt",
            )
        ]
    )
    GPUPreprocessingExperiment(
        name="sicapv2_fsconv_0",
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
                "s3://mlflow/7683e7dbefcd499096fc03ba84353c05/artifacts/model_0.pt",
            )
        ]
    )
    CPUPreprocessingExperiment(
        name="sicapv2_fsconv_rest",
        base="config/preprocessing_sicap_wsi.yml",
        cores=4,
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["params", "link_directory"],
                    [f"v0_low_4000_fsconv_{i}" for i in range(4)],
                ),
                ParameterList(
                    [
                        "pipeline",
                        "stages",
                        3,
                        "feature_extraction",
                        "params",
                        "architecture",
                    ],
                    [
                        f"s3://mlflow/7683e7dbefcd499096fc03ba84353c05/artifacts/model_{i}.pt"
                        for i in range(4)
                    ],
                ),
            ]
        ]
    )
    GPUPreprocessingExperiment(
        name="sicapv2_fsconv_downsample_0",
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
                "s3://mlflow/7683e7dbefcd499096fc03ba84353c05/artifacts/model_0.pt",
            ),
            Parameter(
                [
                    "pipeline",
                    "stages",
                    3,
                    "feature_extraction",
                    "params",
                    "downsample_factor",
                ],
                2.28,
            ),
            Parameter(
                [
                    "pipeline",
                    "stages",
                    3,
                    "feature_extraction",
                    "params",
                    "stride",
                ],
                16,
            ),
        ]
    )
    GPUPreprocessingExperiment(
        name="sicapv2_fsconv_downsample_1",
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
                "s3://mlflow/7683e7dbefcd499096fc03ba84353c05/artifacts/model_1.pt",
            ),
            Parameter(
                [
                    "pipeline",
                    "stages",
                    3,
                    "feature_extraction",
                    "params",
                    "downsample_factor",
                ],
                2.28,
            ),
            Parameter(
                [
                    "pipeline",
                    "stages",
                    3,
                    "feature_extraction",
                    "params",
                    "stride",
                ],
                16,
            ),
        ]
    )
    GPUPreprocessingExperiment(
        name="sicapv2_fsconv_downsample_2",
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
                "s3://mlflow/7683e7dbefcd499096fc03ba84353c05/artifacts/model_2.pt",
            ),
            Parameter(
                [
                    "pipeline",
                    "stages",
                    3,
                    "feature_extraction",
                    "params",
                    "downsample_factor",
                ],
                2.28,
            ),
            Parameter(
                [
                    "pipeline",
                    "stages",
                    3,
                    "feature_extraction",
                    "params",
                    "stride",
                ],
                16,
            ),
        ]
    )
    GPUPreprocessingExperiment(
        name="sicapv2_fsconv_downsample_3",
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
                "s3://mlflow/7683e7dbefcd499096fc03ba84353c05/artifacts/model_3.pt",
            ),
            Parameter(
                [
                    "pipeline",
                    "stages",
                    3,
                    "feature_extraction",
                    "params",
                    "downsample_factor",
                ],
                2.28,
            ),
            Parameter(
                [
                    "pipeline",
                    "stages",
                    3,
                    "feature_extraction",
                    "params",
                    "stride",
                ],
                16,
            ),
        ]
    )
    CPUPreprocessingExperiment(
        name="sicapv2_fsconv_downsample_rest",
        base="config/preprocessing_sicap_wsi.yml",
        cores=4,
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
                    "downsample_factor",
                ],
                2.28,
            ),
            Parameter(
                [
                    "pipeline",
                    "stages",
                    3,
                    "feature_extraction",
                    "params",
                    "stride",
                ],
                16,
            ),
        ],
        sequential=[
            [
                ParameterList(
                    ["params", "link_directory"],
                    [f"v1_low_4000_fsconv_{i}" for i in range(4)],
                ),
                ParameterList(
                    [
                        "pipeline",
                        "stages",
                        3,
                        "feature_extraction",
                        "params",
                        "architecture",
                    ],
                    [
                        f"s3://mlflow/7683e7dbefcd499096fc03ba84353c05/artifacts/model_{i}.pt"
                        for i in range(4)
                    ],
                ),
            ]
        ],
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
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_semi_node_compare",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "data", "augmentation_mode"], "node"),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_semi_graph_compare",
        base="config/sicapv2_wsi_weak2.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "data", "augmentation_mode"], "node"),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    # Rerunning stuff
    GNNTestingExperiment(
        "rerun_for_mlp", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": "_" + x}
                        for x in [
                            "tissue_sicap_pretrained_weighted_node_aug",
                            "tissue_sicap_pretrained_weighted_node_aug",
                            "tissue_sicap_pretrained_weighted_node_aug_keep",
                            "tissue_sicap_pretrained_weighted_node_aug_keep",
                            "tissue_sicap_pretrained_weighted_node_aug_keep",
                            "tissue_sicap_pretrained_weighted_node_aug_keep",
                            "tissue_sicap_resnet_weighted_node_aug_keep",
                            "tissue_sicap_resnet_weighted_node_aug_keep",
                            "tissue_sicap_resnet_weighted_node_aug_keep",
                            "tissue_sicap_resnet_weighted_node_aug_keep",
                            "tissue_sicap_best_log_oversample_keep",
                            "tissue_sicap_resnet_weighted_node_aug",
                            "tissue_sicap_resnet_weighted_node_aug",
                            "tissue_sicap_pretrained_weighted_node_aug",
                            "tissue_sicap_resnet_weighted_node_aug",
                            "tissue_sicap_pretrained_weighted_node_aug",
                            "tissue_sicap_resnet_weighted_node_aug",
                            "tissue_sicap_best_log_oversample_keep",
                            "tissue_sicap_best_log_oversample_keep",
                            "tissue_sicap_best_log_oversample_keep",
                            "tissue_sicap_best_lin_oversample_keep",
                            "tissue_sicap_best_lin_oversample_keep",
                            "tissue_sicap_best_lin_oversample_keep",
                            "tissue_sicap_best_lin_weights_keep",
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
                        "s3://mlflow/650/33aeba7a2a8342339a7c86dee0f1d764/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/96ac1e376be140b39a5ca0c95ca315d9/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/f05480dbcc604e3891d78bf171b5b6eb/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/f4257f7f64bb41a296e544474b3792a2/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/6c2cb3f50b6a457190ab41ca061d10d0/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/40737cb0ac7849e08b242169d08ee775/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/2070652c4e144f3389d01707a2dc4884/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/1de48fdbc66d4a0b8ad6bcace14f8bc1/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/fcec6406e3f84cfea556469ac53cb21d/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/52615d42bcc24984bbdae2f44f10c0ec/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/220b250864b94f9c91b110acfc224607/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/8f41e3ceaa1b4fe79f1d15bdf8692b0e/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/b302a8271fe94b159b600b6989a79502/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/abeaf56ec57748888082838df4127bd1/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/55967fd9ed264237b4b4910d2f4af6c9/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/7a3179a805214f62982fb4100c952121/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/7c8b3e8acf584a01a6eb1c8652971f17/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/36c4b14e69a4469aa9125fc3b4b875a6/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/9ef57d67594249958645ed3a2ee5017a/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/26f53433fa994fb58a4b19737d4e6002/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/39e6b82aeb654f2282f87205d801ffdf/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/8debdd714be240b68575a06f0667dfe1/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/84b90db7d739405a98810c4b333f5c1d/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/131175f5b6a94ff69fa76b3e050b1293/artifacts/best.valid.node.NodeClassificationF1Score",
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
    GNNTestingExperiment(
        "rerun_sicap_semi_node", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": "_" + x}
                        for x in [
                            "semi_sicap_node_0.75",
                            "semi_sicap_node_0.25",
                            "semi_sicap_node_0.75",
                            "semi_sicap_node_0.25",
                            "semi_sicap_node_0.25",
                            "semi_sicap_node_0.25",
                            "semi_sicap_node_0.5",
                            "semi_sicap_node_0.5",
                            "semi_sicap_node_0.5",
                            "semi_sicap_node_0.5",
                            "semi_sicap_node_0.75",
                            "semi_sicap_node_0.75",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/656/e07db15f9fb84e3ba0bc459853fd2c29/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/656/fc2ebe76a7534ba5b47243f16b636433/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/656/bda85b5fa929412faa41fb428ed915db/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/656/f0c3cbc5bfea44d0bdbe328684889302/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/656/880e6edf25c54e7494e1d59c908f8566/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/656/114db3ca2a894a4e9c5ae21d0966a031/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/656/75f7d22e1cc140eb857c479f13efd220/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/656/0f10db91a92b4c13b9f1e2bdedbe9b4b/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/656/0a28d39802314b54867bd541a0b12ba6/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/656/53ac8e23d89f46e5b93432206694399f/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/656/7e9fd04aa2354b75874a6c5d1de139c3/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/656/9e0a4b3261b3432fbc26872685050142/artifacts/best.valid.node.NodeClassificationF1Score",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        "rerun_sicap_semi_graph", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": "_" + x}
                        for x in [
                            "semi_sicap_graph_0.75",
                            "semi_sicap_graph_0.25",
                            "semi_sicap_graph_0.75",
                            "semi_sicap_graph_0.25",
                            "semi_sicap_graph_0.25",
                            "semi_sicap_graph_0.25",
                            "semi_sicap_graph_0.5",
                            "semi_sicap_graph_0.5",
                            "semi_sicap_graph_0.5",
                            "semi_sicap_graph_0.5",
                            "semi_sicap_graph_0.75",
                            "semi_sicap_graph_0.75",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/656/e07db15f9fb84e3ba0bc459853fd2c29/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/656/fc2ebe76a7534ba5b47243f16b636433/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/656/bda85b5fa929412faa41fb428ed915db/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/656/f0c3cbc5bfea44d0bdbe328684889302/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/656/880e6edf25c54e7494e1d59c908f8566/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/656/114db3ca2a894a4e9c5ae21d0966a031/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/656/75f7d22e1cc140eb857c479f13efd220/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/656/0f10db91a92b4c13b9f1e2bdedbe9b4b/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/656/0a28d39802314b54867bd541a0b12ba6/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/656/53ac8e23d89f46e5b93432206694399f/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/656/7e9fd04aa2354b75874a6c5d1de139c3/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                        "s3://mlflow/656/9e0a4b3261b3432fbc26872685050142/artifacts/best.valid.graph.MeanMultiLabelF1Score",
                    ],
                ),
            ]
        ],
    )

    # FSConv Features
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_fsconv_weighted_node_aug",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 512),
            Parameter(["train", "params", "use_weighted_loss"], True),
        ],
        sequential=[
            [
                ParameterList(["train", "data", "fold"], [1, 2, 3, 4]),
                ParameterList(
                    ["train", "data", "graph_directory"],
                    [f"v0_low_4000_fsconv_{i}" for i in range(4)],
                ),
            ]
        ],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_fsconv_weighted_node_aug_keep",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 512),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 200
            ),
        ],
        sequential=[
            [
                ParameterList(["train", "data", "fold"], [1, 2, 3, 4]),
                ParameterList(
                    ["train", "data", "graph_directory"],
                    [f"v0_low_4000_fsconv_{i}" for i in range(4)],
                ),
            ]
        ],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_fsconv_lin_weighted_node_aug",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 512),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], False),
        ],
        sequential=[
            [
                ParameterList(["train", "data", "fold"], [1, 2, 3, 4]),
                ParameterList(
                    ["train", "data", "graph_directory"],
                    [f"v0_low_4000_fsconv_{i}" for i in range(4)],
                ),
            ]
        ],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_fsconv_lin_weighted_node_aug_keep",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 512),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], False),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 200
            ),
        ],
        sequential=[
            [
                ParameterList(["train", "data", "fold"], [1, 2, 3, 4]),
                ParameterList(
                    ["train", "data", "graph_directory"],
                    [f"v0_low_4000_fsconv_{i}" for i in range(4)],
                ),
            ]
        ],
    )

    # Downsampled FSConv Features
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_dfsconv_weighted_node_aug",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 512),
            Parameter(["train", "params", "use_weighted_loss"], True),
        ],
        sequential=[
            [
                ParameterList(["train", "data", "fold"], [1, 2, 3, 4]),
                ParameterList(
                    ["train", "data", "graph_directory"],
                    [f"v1_low_4000_fsconv_{i}" for i in range(4)],
                ),
            ]
        ],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_dfsconv_weighted_node_aug_keep",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 512),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 200
            ),
        ],
        sequential=[
            [
                ParameterList(["train", "data", "fold"], [1, 2, 3, 4]),
                ParameterList(
                    ["train", "data", "graph_directory"],
                    [f"v1_low_4000_fsconv_{i}" for i in range(4)],
                ),
            ]
        ],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_dfsconv_lin_weighted_node_aug",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 512),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], False),
        ],
        sequential=[
            [
                ParameterList(["train", "data", "fold"], [1, 2, 3, 4]),
                ParameterList(
                    ["train", "data", "graph_directory"],
                    [f"v1_low_4000_fsconv_{i}" for i in range(4)],
                ),
            ]
        ],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_dfsconv_lin_weighted_node_aug_keep",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 512),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], False),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], 200
            ),
        ],
        sequential=[
            [
                ParameterList(["train", "data", "fold"], [1, 2, 3, 4]),
                ParameterList(
                    ["train", "data", "graph_directory"],
                    [f"v1_low_4000_fsconv_{i}" for i in range(4)],
                ),
            ]
        ],
    )

    # LR Test Experiments
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_log_os_keep_lr_normal",
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
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_lin_keep_lr_normal",
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
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_log_os_keep_lr_high",
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
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_lin_keep_lr_high",
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
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "data", "augmentation_mode"], ["node"])],
    )

    GNNTestingExperiment(
        "rerun_sicap_for_mlp_new", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": "___" + x}
                        for x in [
                            "tissue_sicap_lin_keep_lr_high",
                            "tissue_sicap_lin_keep_lr_high",
                            "tissue_sicap_lin_keep_lr_high",
                            "tissue_sicap_lin_keep_lr_high",
                            "tissue_sicap_lin_keep_lr_normal",
                            "tissue_sicap_lin_keep_lr_normal",
                            "tissue_sicap_lin_keep_lr_normal",
                            "tissue_sicap_log_os_keep_lr_high",
                            "tissue_sicap_lin_keep_lr_normal",
                            "tissue_sicap_log_os_keep_lr_normal",
                            "tissue_sicap_log_os_keep_lr_high",
                            "tissue_sicap_log_os_keep_lr_high",
                            "tissue_sicap_log_os_keep_lr_high",
                            "tissue_sicap_log_os_keep_lr_normal",
                            "tissue_sicap_log_os_keep_lr_normal",
                            "tissue_sicap_log_os_keep_lr_normal",
                            "tissue_sicap_fsconv_lin_weighted_node_aug_keep",
                            "tissue_sicap_fsconv_lin_weighted_node_aug_keep",
                            "tissue_sicap_fsconv_lin_weighted_node_aug_keep",
                            "tissue_sicap_fsconv_lin_weighted_node_aug_keep",
                            "tissue_sicap_fsconv_lin_weighted_node_aug",
                            "tissue_sicap_fsconv_lin_weighted_node_aug",
                            "tissue_sicap_fsconv_lin_weighted_node_aug",
                            "tissue_sicap_fsconv_lin_weighted_node_aug",
                            "tissue_sicap_fsconv_weighted_node_aug_keep",
                            "tissue_sicap_fsconv_weighted_node_aug_keep",
                            "tissue_sicap_fsconv_weighted_node_aug_keep",
                            "tissue_sicap_fsconv_weighted_node_aug_keep",
                            "tissue_sicap_fsconv_weighted_node_aug",
                            "tissue_sicap_fsconv_weighted_node_aug",
                            "tissue_sicap_fsconv_weighted_node_aug",
                            "tissue_sicap_fsconv_weighted_node_aug",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/650/7ae01ab7bd9544c0abcfae766ad821fd/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/d36ea51bc9df44d2a1901fdfe43152dc/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/74f2a61fd4d54ca38f5603032c4b2d98/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/9520dc23165145629fc40c3ef42a02e5/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/7b7bca2e9f954643b9ef81942a29e774/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/ae39648b700243e4be95cd90a36ad248/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/342eba4b62d54f3188554644a9f003ad/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/b4ee6a6957da440e880282d8db71acfc/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/7c6c682dd1ed4335aaa738b3b740276d/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/66bd6b5196d8417dae786d70bf7b65e9/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/d0436ec060e148f3b7252ae4a73383ea/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/858f8c47303841938da8db718c34c115/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/51b12536c67e445ea3a1c483fb7d691b/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/eb05621879fc4b57a0f85eba3e3013ce/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/5b8baf8dd117457aa98236b46dcd56b3/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/070519da63874e63bccaac6b504b931d/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/d537bfcb17e748b4a75cb12cfed201b1/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/5f5b6b8c0a5b4457aaa2e2c8ce1b1186/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/2ab27ee861eb4e289ba30acc0b82f4a6/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/78d9e9e26fc34860863b7db960e27594/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/5507baf77e6f488c8d6ac0253601b4de/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/a4c0c98a220944f991673fa75ef8a68e/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/d900de459d9049a2b7ec9a27f4e04c07/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/fa8f51e2c8c044fda9e43e8f6df8d941/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/5fc7c53a8c404956b4b304da79e558e8/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/6662c987f46c45a2b2885e88a244804e/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/e1b80947632c4185b19b96aaf11f0912/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/89da49a59ded482fbf02b12a8104dda6/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/b99ea52a0b7a4309a6f27223ed5d025b/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/3a802fc5490f4b19aaed77c75d782aef/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/de75298de1464553be540ad714c91200/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/650/ab3ea737fe094884b7ed93577a41c6f2/artifacts/best.valid.node.segmentation.fF1Score",
                    ],
                ),
            ]
        ],
    )

    run_tags = [
        "tissue_sicap_log_os_keep_lr_normal",
        "tissue_sicap_log_os_keep_lr_normal",
        "tissue_sicap_log_os_keep_lr_normal",
        "tissue_sicap_log_os_keep_lr_normal",
        "tissue_sicap_best_lin_oversample_keep",
        "tissue_sicap_best_lin_oversample_keep",
        "tissue_sicap_best_lin_oversample_keep",
        "tissue_sicap_best_lin_oversample_keep",
        "tissue_sicap_best_weighted_strong_node_aug_keep",
        "tissue_sicap_best_weighted_strong_node_aug_keep",
        "tissue_sicap_best_weighted_strong_node_aug_keep",
        "tissue_sicap_best_weighted_strong_node_aug_keep",
    ]
    run_ids = [
        "s3://mlflow/650/66bd6b5196d8417dae786d70bf7b65e9/artifacts/best.valid.node.segmentation.fF1Score",
        "s3://mlflow/650/eb05621879fc4b57a0f85eba3e3013ce/artifacts/best.valid.node.segmentation.fF1Score",
        "s3://mlflow/650/5b8baf8dd117457aa98236b46dcd56b3/artifacts/best.valid.node.segmentation.fF1Score",
        "s3://mlflow/650/070519da63874e63bccaac6b504b931d/artifacts/best.valid.node.segmentation.fF1Score",
        "s3://mlflow/650/3fc7d90926a1410e92bc2684fbfe505c/artifacts/best.valid.node.segmentation.fF1Score",
        "s3://mlflow/650/39e6b82aeb654f2282f87205d801ffdf/artifacts/best.valid.node.segmentation.fF1Score",
        "s3://mlflow/650/8debdd714be240b68575a06f0667dfe1/artifacts/best.valid.node.segmentation.fF1Score",
        "s3://mlflow/650/84b90db7d739405a98810c4b333f5c1d/artifacts/best.valid.node.segmentation.fF1Score",
        "s3://mlflow/650/c2cadd9218b84e389ebf3b138966edc1/artifacts/best.valid.node.segmentation.fF1Score",
        "s3://mlflow/650/f582548a20ef43e089e022f45780c0f4/artifacts/best.valid.node.segmentation.fF1Score",
        "s3://mlflow/650/ab0b5264549044a79a5ba98d531cd7ca/artifacts/best.valid.node.segmentation.fF1Score",
        "s3://mlflow/650/e59352133a2d4c41abe7f8a241e81b34/artifacts/best.valid.node.segmentation.fF1Score",
    ]
    GNNTestingExperiment(
        "rerun_sicap_threshold_tests", base="config/sicapv2_wsi_strong.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": f"thres_{threshold}_" + x}
                        for x in run_tags
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    run_ids,
                ),
                ParameterList(
                    ["test", "params", "threshold"],
                    [threshold]*len(run_ids)
                )
            ]
        for threshold in [0.001, 0.05, 0.1, 0.15, 0.2, 0.25]],
    )

    # Partial
    CPUPreprocessingExperiment(
        name="sicapv2_wsi_partial",
        base="config/preprocessing_sicap_wsi.yml",
        cores=4,
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(["graph_builders", "params", "partial_annotation"], [50, 25]),
                ParameterList(["params", "partial_annotation"], [50, 25]
                ),
                ParameterList(
                    ["params", "link_directory"],
                    [f"v0_4000_low_partial_{s}" for s in [50, 25]],
                ),
            ]
        ],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_partial_50", base="config/paper_sicap_strong.yml", path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v0_4000_low_partial_50",
            ),
            Parameter(
                ["train", "data", "partial_annotation",
                50]
            )
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_partial_25", base="config/paper_sicap_strong.yml", path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v0_4000_low_partial_25",
            ),
            Parameter(
                ["train", "data", "partial_annotation",
                25]
            )
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="sicap_partial_50", base="config/paper_sicap_semi.yml", path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v0_4000_low_partial_50",
            ),
            Parameter(
                ["train", "data", "partial_annotation",
                25]
            )
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="sicap_partial_25", base="config/paper_sicap_semi.yml", path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v0_4000_low_partial_25",
            ),
            Parameter(
                ["train", "data", "partial_annotation",
                25]
            )
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
