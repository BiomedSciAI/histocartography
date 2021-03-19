import argparse
from pathlib import Path
from typing import Sequence

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

    # Weakly Supervised
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_base", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_base_node", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_scheduler_normal", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
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
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_scheduler_high", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
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
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_insane_dropout", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "model", "graph_classifier_config", "input_dropout"], 0.8
            ),
            Parameter(["train", "model", "gnn_config", "dropout"], 0.8),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_weighted_log", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "params", "use_weighted_loss"], True),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_weighted_lin", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], False),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_no_loc", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_resnet_base", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "graph_directory"], "v0_low_4000_resnet34"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 514),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_resnet_rainbow_lin", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "graph_directory"], "v0_low_4000_resnet34"),
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 512),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], False),
            Parameter(["train", "data", "augmentation_mode"], "node"),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_resnet_rainbow_log", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "graph_directory"], "v0_low_4000_resnet34"),
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 512),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "data", "augmentation_mode"], "node"),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_classifier_dropout", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "data", "augmentation_mode"], "node"),
        ],
        sequential=[
            ParameterList(
                ["train", "model", "graph_classifier_config", "input_dropout"],
                [0.0, 0.2, 0.4, 0.6, 0.8],
            )
        ],
    ) 
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_gnn_depth", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "data", "augmentation_mode"], "node"),
        ],
        sequential=[
            ParameterList(
                ["train", "model", "gnn_config", "n_layers"], [3, 6, 9, 12, 15, 18, 21]
            )
        ],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_rainbox_lin", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "params", "use_log_frequency_weights"], False),
            Parameter(["train", "data", "augmentation_mode"], "node"),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_rainbox_log", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "data", "augmentation_mode"], "node"),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    GNNTestingExperiment(
        name="sicap_rerun_weak", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": "___" + x}
                        for x in [
                            "image_sicap_no_loc",
                            "image_sicap_no_loc",
                            "image_sicap_no_loc",
                            "image_sicap_weighted_lin",
                            "image_sicap_no_loc",
                            "image_sicap_weighted_lin",
                            "image_sicap_weighted_lin",
                            "image_sicap_weighted_lin",
                            "image_sicap_weighted_log",
                            "image_sicap_weighted_log",
                            "image_sicap_weighted_log",
                            "image_sicap_weighted_log",
                            "image_sicap_classifier_dropout",
                            "image_sicap_classifier_dropout",
                            "image_sicap_classifier_dropout",
                            "image_sicap_classifier_dropout",
                            "image_sicap_classifier_dropout",
                            "image_sicap_insane_dropout",
                            "image_sicap_insane_dropout",
                            "image_sicap_insane_dropout",
                            "image_sicap_insane_dropout",
                            "image_sicap_scheduler_high",
                            "image_sicap_scheduler_high",
                            "image_sicap_scheduler_high",
                            "image_sicap_scheduler_high",
                            "image_sicap_scheduler_normal",
                            "image_sicap_scheduler_normal",
                            "image_sicap_scheduler_normal",
                            "image_sicap_scheduler_normal",
                            "image_sicap_base_node",
                            "image_sicap_base_node",
                            "image_sicap_base_node",
                            "image_sicap_base_node",
                            "image_sicap_base",
                            "image_sicap_base",
                            "image_sicap_base",
                            "image_sicap_base",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/652/70472938fddd41328dd60e7ed114b021/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/0fae7982f7ce4f4e8aa1b174b905001a/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/460ae360b0174278b4f67514c0e43876/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/cc9ed9d17cd44f3988b0a09ca2fc5e40/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/dd937a7fba754120a133049f17703aaf/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/1236bfd2f05e48b0a8369291642cdab5/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/4473e975815640a4921f2f3786230245/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/fc425e7b7a814be09154b5e874eadbd3/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/99b8502e0dce4753907f8a4bfacfef15/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/0b07e7584bb84b49992766754c5a1946/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/7262d5816366453181c59adcd996f90d/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/443d29b7155a457aa9edb38ab4ab5326/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/edbbd41fee5e40d08c584a6acfac2b29/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/e893f021d1a34f6f8aa0c82e21334578/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/c58a11b0ca814d3b841adde2b86e0dd0/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/a9bd91fcf3c442bc9aa70f3ee65d33de/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/833a73f5ef874500b91299caacb7781a/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/503e967871cb4c11b0d11b1d88c55410/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/4d4190ce26d74d9d8960a3a0eec43bee/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/7d80ff010b7a46399fde319b4bfb3d7f/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/100d378c4a874d738a45d9f05ec1b720/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/d14606abcdff49b589d0b01fd6365f5f/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/3dea7c0677014ad7bd7b01349f478d8b/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/a28cb84a0dac41549f461c5bf4091a1e/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/2e46f69cebeb41a5b108accecdeda5ef/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/c3abf57991a84fa7870d5d98a67e9562/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/c03e696a948445bb9dff18d35c570142/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/fe83a02e432c440881976541ca33e761/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/e8f71c4585b14df5b4daaf5d894c71ae/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/8cb30bdcbada4a74aa213d8d591f4edf/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/d688e96555344c53ab42e5ecc3831dfa/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/09df1a1a82684bd48816765c0d0f5d94/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/d23044dfb13b4128bd2dc58cc03086db/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/7a6287bad24a4324ba55db2695778c32/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/e991e852f6244460849b55f82061e325/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/826b095fd1f542deb6eb9140c92207af/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/9df65aeda5314a2a99878d74ead48b63/artifacts/best.valid.graph.segmentation.MeanF1Score",
                    ],
                ),
            ]
        ],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_classifier_rainbow_dropout_0.4",
        base="config/sicapv2_wsi_weak.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(
                ["train", "model", "graph_classifier_config", "input_dropout"],
                0.4,
            ),
        ],
        sequential=[
            ParameterList(["train", "data", "fold"], [1, 2, 3, 4]),
        ],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_classifier_rainbow_dropout_0.5",
        base="config/sicapv2_wsi_weak.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(
                ["train", "model", "graph_classifier_config", "input_dropout"],
                0.5,
            ),
        ],
        sequential=[
            ParameterList(["train", "data", "fold"], [1, 2, 3, 4]),
        ],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_classifier_rainbow_dropout_0.3",
        base="config/sicapv2_wsi_weak.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(
                ["train", "model", "graph_classifier_config", "input_dropout"],
                0.3,
            ),
        ],
        sequential=[
            ParameterList(["train", "data", "fold"], [1, 2, 3, 4]),
        ],
    )
    GNNTestingExperiment(
        name="sicap_rerun_weak2", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": "___" + x}
                        for x in [
                            "image_sicap_rainbox_log",
                            "image_sicap_rainbox_log",
                            "image_sicap_rainbox_log",
                            "image_sicap_rainbox_log",
                            "image_sicap_rainbox_lin",
                            "image_sicap_rainbox_lin",
                            "image_sicap_rainbox_lin",
                            "image_sicap_rainbox_lin",
                            "image_sicap_resnet_rainbow_log",
                            "image_sicap_resnet_rainbow_log",
                            "image_sicap_resnet_rainbow_log",
                            "image_sicap_resnet_rainbow_log",
                            "image_sicap_resnet_rainbow_lin",
                            "image_sicap_resnet_rainbow_lin",
                            "image_sicap_resnet_rainbow_lin",
                            "image_sicap_resnet_rainbow_lin",
                            "image_sicap_resnet_base",
                            "image_sicap_resnet_base",
                            "image_sicap_resnet_base",
                            "image_sicap_resnet_base",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/652/7b3670dc3aef4ca087a8f080ecf68b1d/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/1d20edb8aa8a4c36977916c66d9f11db/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/0fb10261195a446a96c2d6825791d7c5/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/9d09b98947254a60bc8c7aff1df5e52f/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/13abe3597c1a4a069cf2a8b0fee4d893/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/25d7c5e0186e453d949aa906c5514226/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/35e0e3dbf378465eb93d44e650e5f187/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/ac3bfd8a01a242d7a7edcb32ecf973c5/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/35d8daae92f54062be8a5427ede54073/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/6436e0ad61c442109a5134317e021e32/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/e889ed9d66074779bfc8e5a1277e6b44/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/79d4c77f8ab0494883eb6a34c7b8c9fb/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/6cc883c983b241aeb894e5b6c96411cc/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/3398cb45521046d5bb2fdb594d5ff6b2/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/f9587e9ce709431197afb772fdda5ecf/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/8dc5d67df56d4f9b9a897c6e2d162fd4/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/722da1dcd3034ebf8ab45c8d98249b51/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/4fe5e96a745b42d0977d5b6bf75d9477/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/59e458d7acf9494a8fa22c4294dd19ca/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/092ae210266b4be5ae585e4210739a35/artifacts/best.valid.graph.segmentation.MeanF1Score",
                    ],
                ),
            ]
        ],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_high_dropout", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "model", "graph_classifier_config", "input_dropout"], 0.7
            ),
            Parameter(["train", "model", "gnn_config", "dropout"], 0.7),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_gnn_6", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "model", "gnn_config", "n_layers"], 6),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_gnn_9", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "model", "gnn_config", "n_layers"], 9),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_gnn_15", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "model", "gnn_config", "n_layers"], 15),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_lstm", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "model", "gnn_config", "agg_operator"], "lstm"),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_none", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "model", "gnn_config", "agg_operator"], "none"),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    StronglySupervisedGraphClassificationExperiment(
        name="sicap_rep_final",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    GNNTestingExperiment(
        name="sicap_rerun_weak_fixed", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "image_sicap_classifier_rainbow_dropout_0.5",
                            "image_sicap_classifier_rainbow_dropout_0.5",
                            "image_sicap_classifier_rainbow_dropout_0.5",
                            "image_sicap_classifier_rainbow_dropout_0.5",
                            "image_sicap_classifier_rainbow_dropout_0.3",
                            "image_sicap_classifier_rainbow_dropout_0.3",
                            "image_sicap_classifier_rainbow_dropout_0.3",
                            "image_sicap_classifier_rainbow_dropout_0.3",
                            "image_sicap_classifier_rainbow_dropout_0.4",
                            "image_sicap_classifier_rainbow_dropout_0.4",
                            "image_sicap_classifier_rainbow_dropout_0.4",
                            "image_sicap_classifier_rainbow_dropout_0.4",
                            "image_sicap_rainbox_log",
                            "image_sicap_rainbox_log",
                            "image_sicap_rainbox_log",
                            "image_sicap_rainbox_log",
                            "image_sicap_rainbox_lin",
                            "image_sicap_rainbox_lin",
                            "image_sicap_rainbox_lin",
                            "image_sicap_rainbox_lin",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/652/199ef1058de6488d84a29c3f4d92536d/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/ea69cc6930104c499cbd3f4adb3cc8fb/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/a1e1721f37114bbdb08f59e27c704783/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/2b0bcea0f4fb4a3d95ce35928393dbd3/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/b664f1e4829d43ad8d4d31933e68f105/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/44e197e92ccb4c3f8d5865c664bfed85/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/7a9837554a854c009e938fd77ef5474b/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/0abeca81494841058077a201f348e097/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/ac139da85bae4202b934facd24ba4b4d/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/38792a2fec31405bb970c5c5e2d959a4/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/a8a579cd11654d95937627227b04d5b3/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/d83e3b1329654f37885d144618cb456a/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/ed6ebf42ffbf4ba2b327b515495304dd/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/f6f340d4c06a47e6bd01bf727eae5a92/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/59820994cc234f5a9ce2653f271335ed/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/4c6b8b04660546f18cead1419f5eff9b/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/88a01fabee5048afb95ec6a3a243dfcd/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/fb78ef90608a4a84b7c1d2b7a0a666c4/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/ae3f00895a114fc2a606c82140fd2aa0/artifacts/best.valid.graph.segmentation.MeanF1Score",
                        "s3://mlflow/652/03afc604d7f84a928ddeff215aec3167/artifacts/best.valid.graph.segmentation.MeanF1Score",
                    ],
                ),
            ]
        ],
    )

    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_new_node", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_new_node_lr", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.001,
            ),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_new_node_drop", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(
                ["train", "model", "graph_classifier_config", "input_dropout"], 0.7
            ),
            Parameter(["train", "model", "gnn_config", "dropout"], 0.7),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    StronglySupervisedGraphClassificationExperiment(
        name="sicap_rep_final_lr",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_rep_final_again",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_new_node_again", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    GNNTestingExperiment(
        name="sicap_rerun_partial_node_best",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_sicap_partial_25_node_best",
                            "semi_sicap_partial_25_node_best",
                            "semi_sicap_partial_25_node_best",
                            "semi_sicap_partial_25_node_best",
                            "semi_sicap_partial_50_node_best",
                            "semi_sicap_partial_50_node_best",
                            "semi_sicap_partial_50_node_best",
                            "semi_sicap_partial_50_node_best",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/656/613cc87318454a1d836d1bb2e3ef1a10/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/ddb10a451f41417c99d41ecb681c67e5/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/399d0859f9fd4963b7e7d053f1b21cd0/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/aa4ed943a6b44be4b4c839dbb3dccb82/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/3651062d5cb64c688469518bec6b52fa/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/89d64f8c12ef4afba3254e88bee79310/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/631c9a664a414cbe8d2ae1c084c48ff2/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/c2cb7839fce3446b8aa2f3d6a577930b/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="sicap_rerun_partial_graph_proxy",
        base="config/sicapv2_wsi_weak.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_sicap_partial_25_graph_proxy",
                            "semi_sicap_partial_25_graph_proxy",
                            "semi_sicap_partial_25_graph_proxy",
                            "semi_sicap_partial_25_graph_proxy",
                            "semi_sicap_partial_50_graph_proxy",
                            "semi_sicap_partial_50_graph_proxy",
                            "semi_sicap_partial_50_graph_proxy",
                            "semi_sicap_partial_50_graph_proxy",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/656/613cc87318454a1d836d1bb2e3ef1a10/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/ddb10a451f41417c99d41ecb681c67e5/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/399d0859f9fd4963b7e7d053f1b21cd0/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/aa4ed943a6b44be4b4c839dbb3dccb82/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/3651062d5cb64c688469518bec6b52fa/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/89d64f8c12ef4afba3254e88bee79310/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/631c9a664a414cbe8d2ae1c084c48ff2/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/c2cb7839fce3446b8aa2f3d6a577930b/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="sicap_rerun_partial_node_proxy",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_sicap_partial_25_node_proxy",
                            "semi_sicap_partial_25_node_proxy",
                            "semi_sicap_partial_25_node_proxy",
                            "semi_sicap_partial_25_node_proxy",
                            "semi_sicap_partial_50_node_proxy",
                            "semi_sicap_partial_50_node_proxy",
                            "semi_sicap_partial_50_node_proxy",
                            "semi_sicap_partial_50_node_proxy",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/656/613cc87318454a1d836d1bb2e3ef1a10/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/ddb10a451f41417c99d41ecb681c67e5/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/399d0859f9fd4963b7e7d053f1b21cd0/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/aa4ed943a6b44be4b4c839dbb3dccb82/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/3651062d5cb64c688469518bec6b52fa/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/89d64f8c12ef4afba3254e88bee79310/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/631c9a664a414cbe8d2ae1c084c48ff2/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/c2cb7839fce3446b8aa2f3d6a577930b/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="sicap_rerun_partial_graph_best",
        base="config/sicapv2_wsi_weak.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_sicap_partial_25_graph_best",
                            "semi_sicap_partial_25_graph_best",
                            "semi_sicap_partial_25_graph_best",
                            "semi_sicap_partial_25_graph_best",
                            "semi_sicap_partial_50_graph_best",
                            "semi_sicap_partial_50_graph_best",
                            "semi_sicap_partial_50_graph_best",
                            "semi_sicap_partial_50_graph_best",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/656/613cc87318454a1d836d1bb2e3ef1a10/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/ddb10a451f41417c99d41ecb681c67e5/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/399d0859f9fd4963b7e7d053f1b21cd0/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/aa4ed943a6b44be4b4c839dbb3dccb82/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/3651062d5cb64c688469518bec6b52fa/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/89d64f8c12ef4afba3254e88bee79310/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/631c9a664a414cbe8d2ae1c084c48ff2/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/c2cb7839fce3446b8aa2f3d6a577930b/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                    ],
                ),
            ]
        ],
    )

    # Node dropout
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_node_dropout_0.05", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "data", "node_dropout"], 0.05),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_node_dropout_0.1", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "data", "node_dropout"], 0.1),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_node_dropout_0.2", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "data", "node_dropout"], 0.2),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_node_dropout_0.3", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "data", "node_dropout"], 0.3),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_node_dropout_0.4", base="config/sicapv2_wsi_weak.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "data", "node_dropout"], 0.4),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    GNNTestingExperiment(
        name="sicap_image_rerun_failed",
        base="config/sicapv2_wsi_weak.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "image_sicap_new_node_MeanDatasetDice",
                            "image_sicap_new_node_GleasonScoreF1",
                            "image_sicap_new_node_MeanDatasetDice",
                            "image_sicap_new_node_GleasonScoreF1",
                            "image_sicap_new_node_MeanDatasetDice",
                            "image_sicap_new_node_GleasonScoreF1",
                            "image_sicap_new_node_MeanDatasetDice",
                            "image_sicap_new_node_GleasonScoreF1",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/652/45cf249682894b5ebb8879cf2df2a6e9/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/45cf249682894b5ebb8879cf2df2a6e9/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                        "s3://mlflow/652/b544717e221c48958159347518226f48/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/b544717e221c48958159347518226f48/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                        "s3://mlflow/652/c00bfde5309248acbcb2bfb31721ab63/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/c00bfde5309248acbcb2bfb31721ab63/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                        "s3://mlflow/652/a0d4a8c4a62d4b1e9a9905ff7f6351c4/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/a0d4a8c4a62d4b1e9a9905ff7f6351c4/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="sicap_tissue_rerun_failed",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_sicap_rep_final_again_MeanDatasetDice",
                            "tissue_sicap_rep_final_again_GleasonScoreF1",
                            "tissue_sicap_rep_final_again_MeanDatasetDice",
                            "tissue_sicap_rep_final_again_GleasonScoreF1",
                            "tissue_sicap_rep_final_again_MeanDatasetDice",
                            "tissue_sicap_rep_final_again_GleasonScoreF1",
                            "tissue_sicap_rep_final_again_MeanDatasetDice",
                            "tissue_sicap_rep_final_again_GleasonScoreF1",
                            "tissue_sicap_partial_100_MeanDatasetDice",
                            "tissue_sicap_partial_100_GleasonScoreF1",
                            "tissue_sicap_partial_100_MeanDatasetDice",
                            "tissue_sicap_partial_100_GleasonScoreF1",
                            "tissue_sicap_partial_100_MeanDatasetDice",
                            "tissue_sicap_partial_100_GleasonScoreF1",
                            "tissue_sicap_partial_100_MeanDatasetDice",
                            "tissue_sicap_partial_100_GleasonScoreF1",
                            "tissue_sicap_rep_final2_again_MeanDatasetDice",
                            "tissue_sicap_rep_final2_again_GleasonScoreF1",
                            "tissue_sicap_rep_final2_again_MeanDatasetDice",
                            "tissue_sicap_rep_final2_again_GleasonScoreF1",
                            "tissue_sicap_rep_final2_again_MeanDatasetDice",
                            "tissue_sicap_rep_final2_again_GleasonScoreF1",
                            "tissue_sicap_rep_final2_again_MeanDatasetDice",
                            "tissue_sicap_rep_final2_again_GleasonScoreF1",
                            "tissue_sicap_rep_final_lr_MeanDatasetDice",
                            "tissue_sicap_rep_final_lr_GleasonScoreF1",
                            "tissue_sicap_rep_final_lr_MeanDatasetDice",
                            "tissue_sicap_rep_final_lr_GleasonScoreF1",
                            "tissue_sicap_rep_final_lr_MeanDatasetDice",
                            "tissue_sicap_rep_final_lr_GleasonScoreF1",
                            "tissue_sicap_rep_final_lr_MeanDatasetDice",
                            "tissue_sicap_rep_final_lr_GleasonScoreF1",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/650/7cdaf98efe58425491b17a50b9f6fa43/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/7cdaf98efe58425491b17a50b9f6fa43/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/2ec7234baca24daeb2ba27d12bdd92af/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/2ec7234baca24daeb2ba27d12bdd92af/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/b00531ab6d7c4a9d8ae8e7fd2fd591ac/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/b00531ab6d7c4a9d8ae8e7fd2fd591ac/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/a46267c70578434f8fff20df89b55147/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/a46267c70578434f8fff20df89b55147/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/d2f5fdb4a8b14aa88551910ffdf2ae51/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/d2f5fdb4a8b14aa88551910ffdf2ae51/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/1153714fe1af456fb794c70356347e05/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/1153714fe1af456fb794c70356347e05/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/4313b9d353d94e7aa99418ae59f16537/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/4313b9d353d94e7aa99418ae59f16537/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/eb91ed4f7b6245c49d0ebaf9420eef02/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/eb91ed4f7b6245c49d0ebaf9420eef02/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/ee40f49562a847a98a5ba9688aefb244/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/ee40f49562a847a98a5ba9688aefb244/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/00011823628642169908d3636b142151/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/00011823628642169908d3636b142151/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/2ed35bb52fe14706a622986cdb452826/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/2ed35bb52fe14706a622986cdb452826/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/318a1695d23f498b8189bcfb013bb892/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/318a1695d23f498b8189bcfb013bb892/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/5a7dd480a8fb48cf94868d50d03ea447/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/5a7dd480a8fb48cf94868d50d03ea447/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/27f250a920a9419a9bca303b3554e4be/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/27f250a920a9419a9bca303b3554e4be/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/b7297c5d0d1e4d92a77f115f0e2d271a/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/b7297c5d0d1e4d92a77f115f0e2d271a/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/e07d11ddc4f446e3a5ed3e4b0bb78d78/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/e07d11ddc4f446e3a5ed3e4b0bb78d78/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="sicap_semi_node_rerun_failed",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_sicap_partial_100_node",
                            "semi_sicap_partial_100_node",
                            "semi_sicap_partial_100_node",
                            "semi_sicap_partial_100_node",
                            "semi_sicap_partial_25_node",
                            "semi_sicap_partial_25_node",
                            "semi_sicap_partial_25_node",
                            "semi_sicap_partial_25_node",
                            "semi_sicap_partial_50_node",
                            "semi_sicap_partial_50_node",
                            "semi_sicap_partial_50_node",
                            "semi_sicap_partial_50_node",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/656/5e6b5e1ac6ca4122bda18c4ff75d3d51/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/5ec6a86117d44bceb2d58ab7585c0db8/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/f5409b4fb8ba486d80af6835c27e5648/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/d072f58266c24f6eb0f92c699d79896f/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/613cc87318454a1d836d1bb2e3ef1a10/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/ddb10a451f41417c99d41ecb681c67e5/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/399d0859f9fd4963b7e7d053f1b21cd0/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/aa4ed943a6b44be4b4c839dbb3dccb82/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/3651062d5cb64c688469518bec6b52fa/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/89d64f8c12ef4afba3254e88bee79310/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/631c9a664a414cbe8d2ae1c084c48ff2/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/c2cb7839fce3446b8aa2f3d6a577930b/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="sicap_semi_graph_rerun_failed",
        base="config/sicapv2_wsi_weak.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_sicap_partial_100_graph",
                            "semi_sicap_partial_100_graph",
                            "semi_sicap_partial_100_graph",
                            "semi_sicap_partial_100_graph",
                            "semi_sicap_partial_25_graph",
                            "semi_sicap_partial_25_graph",
                            "semi_sicap_partial_25_graph",
                            "semi_sicap_partial_25_graph",
                            "semi_sicap_partial_50_graph",
                            "semi_sicap_partial_50_graph",
                            "semi_sicap_partial_50_graph",
                            "semi_sicap_partial_50_graph",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/656/5e6b5e1ac6ca4122bda18c4ff75d3d51/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/5ec6a86117d44bceb2d58ab7585c0db8/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/f5409b4fb8ba486d80af6835c27e5648/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/d072f58266c24f6eb0f92c699d79896f/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/613cc87318454a1d836d1bb2e3ef1a10/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/ddb10a451f41417c99d41ecb681c67e5/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/399d0859f9fd4963b7e7d053f1b21cd0/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/aa4ed943a6b44be4b4c839dbb3dccb82/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/3651062d5cb64c688469518bec6b52fa/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/89d64f8c12ef4afba3254e88bee79310/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/631c9a664a414cbe8d2ae1c084c48ff2/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/c2cb7839fce3446b8aa2f3d6a577930b/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="sicap_tissue_rerun_missing",
        base="config/sicapv2_wsi_strong.yml",
        queue="dev",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_sicap_rep_final_MeanDatasetDice",
                            "tissue_sicap_rep_final_GleasonScoreF1",
                            "tissue_sicap_rep_final_MeanDatasetDice",
                            "tissue_sicap_rep_final_GleasonScoreF1",
                            "tissue_sicap_rep_final_MeanDatasetDice",
                            "tissue_sicap_rep_final_GleasonScoreF1",
                            "tissue_sicap_rep_final_MeanDatasetDice",
                            "tissue_sicap_rep_final_GleasonScoreF1",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/650/bb0056a5a90745189bbbfcde5b42b447/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/bb0056a5a90745189bbbfcde5b42b447/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/3f6dfe4bebf94a879049cdaa2bb32ec7/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/3f6dfe4bebf94a879049cdaa2bb32ec7/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/f46b2078365a438bac5f1ca00d7f67e3/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/f46b2078365a438bac5f1ca00d7f67e3/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/cff6b265bae1494a823e5715ecb92925/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/cff6b265bae1494a823e5715ecb92925/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                    ],
                ),
            ]
        ],
    )

    SemiSupervisedGraphClassificationExperiment(
        name="sicap_semi_100",
        base="config/paper_sicap_semi.yml",
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

    GNNTestingExperiment(
        name="sicap_tissue_rerun_loss",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_sicap_rep_final_again_final2_loss",
                            "tissue_sicap_rep_final_again_final2_loss",
                            "tissue_sicap_rep_final_again_final2_loss",
                            "tissue_sicap_rep_final_again_final2_loss",
                            "tissue_sicap_rep_final_again_final_loss",
                            "tissue_sicap_rep_final_again_final_loss",
                            "tissue_sicap_rep_final_again_final_loss",
                            "tissue_sicap_rep_final_again_final_loss",
                            "tissue_sicap_rep_final_final_loss",
                            "tissue_sicap_rep_final_final_loss",
                            "tissue_sicap_rep_final_final_loss",
                            "tissue_sicap_rep_final_final_loss",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/650/7cdaf98efe58425491b17a50b9f6fa43/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/2ec7234baca24daeb2ba27d12bdd92af/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/b00531ab6d7c4a9d8ae8e7fd2fd591ac/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/a46267c70578434f8fff20df89b55147/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/ee40f49562a847a98a5ba9688aefb244/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/00011823628642169908d3636b142151/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/2ed35bb52fe14706a622986cdb452826/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/318a1695d23f498b8189bcfb013bb892/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/bb0056a5a90745189bbbfcde5b42b447/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/3f6dfe4bebf94a879049cdaa2bb32ec7/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/f46b2078365a438bac5f1ca00d7f67e3/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/cff6b265bae1494a823e5715ecb92925/artifacts/best.valid.node.loss",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="sicap_tissue_rerun_f1",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_sicap_rep_final_again_final2_f1",
                            "tissue_sicap_rep_final_again_final2_f1",
                            "tissue_sicap_rep_final_again_final2_f1",
                            "tissue_sicap_rep_final_again_final2_f1",
                            "tissue_sicap_rep_final_again_final_f1",
                            "tissue_sicap_rep_final_again_final_f1",
                            "tissue_sicap_rep_final_again_final_f1",
                            "tissue_sicap_rep_final_again_final_f1",
                            "tissue_sicap_rep_final_final_f1",
                            "tissue_sicap_rep_final_final_f1",
                            "tissue_sicap_rep_final_final_f1",
                            "tissue_sicap_rep_final_final_f1",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/650/7cdaf98efe58425491b17a50b9f6fa43/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/2ec7234baca24daeb2ba27d12bdd92af/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/b00531ab6d7c4a9d8ae8e7fd2fd591ac/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/a46267c70578434f8fff20df89b55147/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/ee40f49562a847a98a5ba9688aefb244/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/00011823628642169908d3636b142151/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/2ed35bb52fe14706a622986cdb452826/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/318a1695d23f498b8189bcfb013bb892/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/bb0056a5a90745189bbbfcde5b42b447/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/3f6dfe4bebf94a879049cdaa2bb32ec7/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/f46b2078365a438bac5f1ca00d7f67e3/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/650/cff6b265bae1494a823e5715ecb92925/artifacts/best.valid.node.NodeClassificationF1Score",
                    ],
                ),
            ]
        ],
    )

    StronglySupervisedGraphClassificationExperiment(
        name="sicap_previous_best",
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

    GNNTestingExperiment(
        name="sicap_tissue_partial_final_loss",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_sicap_partial_25_final_loss",
                            "tissue_sicap_partial_25_final_loss",
                            "tissue_sicap_partial_25_final_loss",
                            "tissue_sicap_partial_25_final_loss",
                            "tissue_sicap_partial_50_final_loss",
                            "tissue_sicap_partial_50_final_loss",
                            "tissue_sicap_partial_50_final_loss",
                            "tissue_sicap_partial_50_final_loss",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/650/c259def9493945878ee1f4ee48ea2159/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/b509b5f8d8dd4e3490a5fdf168865c88/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/dd55ae36f02e4ad9abf682a3772cc213/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/0f4bfe1e2afb48ef8d9e5d7141bad73f/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/473d9cb1a2064562b1b09fc6e8c06591/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/31e4ce2f76004b1cb940a1a79e849ed3/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/7b9ac942454947ae84ec923d60f0fcba/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/c049ad0da005467b8f86ffee24d25b46/artifacts/best.valid.node.loss",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="sicap_tissue_partial_final_dice",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_sicap_partial_25_final_dice",
                            "tissue_sicap_partial_25_final_dice",
                            "tissue_sicap_partial_25_final_dice",
                            "tissue_sicap_partial_25_final_dice",
                            "tissue_sicap_partial_50_final_dice",
                            "tissue_sicap_partial_50_final_dice",
                            "tissue_sicap_partial_50_final_dice",
                            "tissue_sicap_partial_50_final_dice",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/650/c259def9493945878ee1f4ee48ea2159/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/b509b5f8d8dd4e3490a5fdf168865c88/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/dd55ae36f02e4ad9abf682a3772cc213/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/0f4bfe1e2afb48ef8d9e5d7141bad73f/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/473d9cb1a2064562b1b09fc6e8c06591/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/31e4ce2f76004b1cb940a1a79e849ed3/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/7b9ac942454947ae84ec923d60f0fcba/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/c049ad0da005467b8f86ffee24d25b46/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="sicap_semi_partial_final_node_loss",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_sicap_partial_25_final_loss",
                            "semi_sicap_partial_25_final_loss",
                            "semi_sicap_partial_25_final_loss",
                            "semi_sicap_partial_25_final_loss",
                            "semi_sicap_partial_50_final_loss",
                            "semi_sicap_partial_50_final_loss",
                            "semi_sicap_partial_50_final_loss",
                            "semi_sicap_partial_50_final_loss",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/656/613cc87318454a1d836d1bb2e3ef1a10/artifacts/best.valid.node.loss",
                        "s3://mlflow/656/ddb10a451f41417c99d41ecb681c67e5/artifacts/best.valid.node.loss",
                        "s3://mlflow/656/399d0859f9fd4963b7e7d053f1b21cd0/artifacts/best.valid.node.loss",
                        "s3://mlflow/656/aa4ed943a6b44be4b4c839dbb3dccb82/artifacts/best.valid.node.loss",
                        "s3://mlflow/656/3651062d5cb64c688469518bec6b52fa/artifacts/best.valid.node.loss",
                        "s3://mlflow/656/89d64f8c12ef4afba3254e88bee79310/artifacts/best.valid.node.loss",
                        "s3://mlflow/656/631c9a664a414cbe8d2ae1c084c48ff2/artifacts/best.valid.node.loss",
                        "s3://mlflow/656/c2cb7839fce3446b8aa2f3d6a577930b/artifacts/best.valid.node.loss",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="sicap_semi_partial_final_node_dice",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_sicap_partial_25_final_dice",
                            "semi_sicap_partial_25_final_dice",
                            "semi_sicap_partial_25_final_dice",
                            "semi_sicap_partial_25_final_dice",
                            "semi_sicap_partial_50_final_dice",
                            "semi_sicap_partial_50_final_dice",
                            "semi_sicap_partial_50_final_dice",
                            "semi_sicap_partial_50_final_dice",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/656/613cc87318454a1d836d1bb2e3ef1a10/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/ddb10a451f41417c99d41ecb681c67e5/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/399d0859f9fd4963b7e7d053f1b21cd0/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/aa4ed943a6b44be4b4c839dbb3dccb82/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/3651062d5cb64c688469518bec6b52fa/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/89d64f8c12ef4afba3254e88bee79310/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/631c9a664a414cbe8d2ae1c084c48ff2/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/c2cb7839fce3446b8aa2f3d6a577930b/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                    ],
                ),
            ]
        ],
    )

    GNNTestingExperiment(
        name="sicap_image_non_stuck",
        base="config/sicapv2_wsi_weak.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "image_sicap_new_node_again_final_dice",
                            "image_sicap_new_node_again_final_dice",
                            "image_sicap_new_node_again_final_dice",
                            "image_sicap_new_node_again_final_dice",
                            "image_sicap_new_node_again_final_f1",
                            "image_sicap_new_node_again_final_f1",
                            "image_sicap_new_node_again_final_f1",
                            "image_sicap_new_node_again_final_f1",
                            "image_sicap_new_node_again_final_loss",
                            "image_sicap_new_node_again_final_loss",
                            "image_sicap_new_node_again_final_loss",
                            "image_sicap_new_node_again_final_loss",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/652/eb1bdd2226a04cfca0fa807f7aac143d/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/801193cf8174488384e074e95c04d4b6/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/f37bdeb598d04b6bb4479c55a67715dc/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/4fa070b37423402ca20fe4a42b56f794/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/eb1bdd2226a04cfca0fa807f7aac143d/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                        "s3://mlflow/652/801193cf8174488384e074e95c04d4b6/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                        "s3://mlflow/652/f37bdeb598d04b6bb4479c55a67715dc/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                        "s3://mlflow/652/4fa070b37423402ca20fe4a42b56f794/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                        "s3://mlflow/652/eb1bdd2226a04cfca0fa807f7aac143d/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/801193cf8174488384e074e95c04d4b6/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/f37bdeb598d04b6bb4479c55a67715dc/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/4fa070b37423402ca20fe4a42b56f794/artifacts/best.valid.graph.loss",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="sicap_image_non_stuck_dropout",
        base="config/sicapv2_wsi_weak.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "image_sicap_node_dropout_0.1_final_loss",
                            "image_sicap_node_dropout_0.1_final_loss",
                            "image_sicap_node_dropout_0.1_final_loss",
                            "image_sicap_node_dropout_0.1_final_loss",
                            "image_sicap_node_dropout_0.1_final_dice",
                            "image_sicap_node_dropout_0.1_final_dice",
                            "image_sicap_node_dropout_0.1_final_dice",
                            "image_sicap_node_dropout_0.1_final_dice",
                            "image_sicap_node_dropout_0.1_final_f1",
                            "image_sicap_node_dropout_0.1_final_f1",
                            "image_sicap_node_dropout_0.1_final_f1",
                            "image_sicap_node_dropout_0.1_final_f1",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/652/e0c22800d8984730bb4ab8a3b06cc442/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/87cd67c32d4443f0b95d114383c8c5fe/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/49f77c26246449a09d05725d0226a3f5/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/014efaf26ec64382931498bd81838fec/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/e0c22800d8984730bb4ab8a3b06cc442/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/87cd67c32d4443f0b95d114383c8c5fe/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/49f77c26246449a09d05725d0226a3f5/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/014efaf26ec64382931498bd81838fec/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/e0c22800d8984730bb4ab8a3b06cc442/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                        "s3://mlflow/652/87cd67c32d4443f0b95d114383c8c5fe/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                        "s3://mlflow/652/49f77c26246449a09d05725d0226a3f5/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                        "s3://mlflow/652/014efaf26ec64382931498bd81838fec/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                    ],
                ),
            ]
        ],
    )

    GNNTestingExperiment(
        name="sicap_image_gnn_layers",
        base="config/sicapv2_wsi_weak.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "image_sicap_gnn_15_final",
                            "image_sicap_gnn_15_final_loss",
                            "image_sicap_gnn_15_final_loss",
                            "image_sicap_gnn_15_final_loss",
                            "image_sicap_gnn_9_final_loss",
                            "image_sicap_gnn_9_final_loss",
                            "image_sicap_gnn_9_final_loss",
                            "image_sicap_gnn_9_final_loss",
                            "image_sicap_gnn_6_final_loss",
                            "image_sicap_gnn_6_final_loss",
                            "image_sicap_gnn_6_final_loss",
                            "image_sicap_gnn_6_final_loss",
                            "image_sicap_gnn_15_final_dice",
                            "image_sicap_gnn_15_final_dice",
                            "image_sicap_gnn_15_final_dice",
                            "image_sicap_gnn_15_final_dice",
                            "image_sicap_gnn_9_final_dice",
                            "image_sicap_gnn_9_final_dice",
                            "image_sicap_gnn_9_final_dice",
                            "image_sicap_gnn_9_final_dice",
                            "image_sicap_gnn_6_final_dice",
                            "image_sicap_gnn_6_final_dice",
                            "image_sicap_gnn_6_final_dice",
                            "image_sicap_gnn_6_final_dice",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/652/b542fbd6693e46d09cfa2d234663de9d/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/c72a08b2edc3471598f59fe3c724b1a9/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/c21d13ade7334566b95fc6d0642bbb6a/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/f7e75cee27524caca884c58b8f1681f6/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/08271e1b46674121a4c2e6d3410055e0/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/e80d2317be624ebc823a8692f7093b1b/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/2a8c6556a20543ba844e2e5afa71e1fe/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/d87ffac10f5b4271b3c2baa085a8ef90/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/70dcd6a4ed364b7fbcd3d5a1e3004aec/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/b65e864848cd4fbc8aaf4d161150347a/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/5ddd991e516341c8a1fe46d08c0667d1/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/54b5aae1a26c488889c64c4d7b85523f/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/b542fbd6693e46d09cfa2d234663de9d/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/c72a08b2edc3471598f59fe3c724b1a9/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/c21d13ade7334566b95fc6d0642bbb6a/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/f7e75cee27524caca884c58b8f1681f6/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/08271e1b46674121a4c2e6d3410055e0/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/e80d2317be624ebc823a8692f7093b1b/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/2a8c6556a20543ba844e2e5afa71e1fe/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/d87ffac10f5b4271b3c2baa085a8ef90/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/70dcd6a4ed364b7fbcd3d5a1e3004aec/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/b65e864848cd4fbc8aaf4d161150347a/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/5ddd991e516341c8a1fe46d08c0667d1/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/54b5aae1a26c488889c64c4d7b85523f/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                    ],
                ),
            ]
        ],
    )

    GNNTestingExperiment(
        name="sicap_semi_100",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_sicap_semi_100_final_loss",
                            "semi_sicap_semi_100_final_loss",
                            "semi_sicap_semi_100_final_loss",
                            "semi_sicap_semi_100_final_loss",
                            "semi_sicap_semi_100_final_nodef1",
                            "semi_sicap_semi_100_final_nodef1",
                            "semi_sicap_semi_100_final_nodef1",
                            "semi_sicap_semi_100_final_nodef1",
                            "semi_sicap_semi_100_final_dice",
                            "semi_sicap_semi_100_final_dice",
                            "semi_sicap_semi_100_final_dice",
                            "semi_sicap_semi_100_final_dice",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/656/ba94f270a23a475fa4bd7845fd2eea25/artifacts/best.valid.node.loss",
                        "s3://mlflow/656/3566ad857fc74ebea41e792d189709ed/artifacts/best.valid.node.loss",
                        "s3://mlflow/656/fb51cf8b926047ff831825f5c765d968/artifacts/best.valid.node.loss",
                        "s3://mlflow/656/bdecb71112974f039ba452e00a29fc84/artifacts/best.valid.node.loss",
                        "s3://mlflow/656/ba94f270a23a475fa4bd7845fd2eea25/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/656/3566ad857fc74ebea41e792d189709ed/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/656/fb51cf8b926047ff831825f5c765d968/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/656/bdecb71112974f039ba452e00a29fc84/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/656/ba94f270a23a475fa4bd7845fd2eea25/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/3566ad857fc74ebea41e792d189709ed/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/fb51cf8b926047ff831825f5c765d968/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/bdecb71112974f039ba452e00a29fc84/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                    ],
                ),
            ]
        ],
    )

    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_dropout_0.05",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "data", "node_dropout"], 0.05),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], None
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_dropout_0.1",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "data", "node_dropout"], 0.1),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], None
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_dropout_0.2",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "data", "node_dropout"], 0.2),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], None
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_dropout_0.3",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "data", "node_dropout"], 0.3),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], None
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_dropout_0.4",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "data", "node_dropout"], 0.4),
            Parameter(
                ["train", "params", "loss", "node", "params", "nodes_to_keep"], None
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_gnn_15",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "model", "gnn_config", "n_layers"], 15),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_gnn_9",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "model", "gnn_config", "n_layers"], 9),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_gnn_12",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "model", "gnn_config", "n_layers"], 12),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_gnn_3",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "model", "gnn_config", "n_layers"], 3),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_paper_image_6_layer",
        base="config/paper_sicap_weak.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "model", "gnn_config", "n_layers"], 6),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_with_cat",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "cat"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1282),
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_paper_image_6_layer_no",
        base="config/paper_sicap_weak.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "model", "gnn_config", "n_layers"], 6),
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    CPUPreprocessingExperiment(
        name="sicapv2_wsi_partial_10",
        base="config/preprocessing_sicap_wsi.yml",
        cores=4,
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(["graph_builders", "params", "partial_annotation"], [10]),
                ParameterList(["params", "partial_annotation"], [10]),
                ParameterList(
                    ["params", "link_directory"],
                    [f"v0_4000_low_partial_{s}" for s in [10]],
                ),
            ]
        ],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_partial_10",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "v0_4000_low_partial_10",
            ),
            Parameter(["train", "data", "partial_annotation"], 10),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="sicap_partial_10",
        base="config/paper_sicap_semi.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "v0_4000_low_partial_10",
            ),
            Parameter(["train", "data", "partial_annotation"], 10),
        ],
        sequential=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    GNNTestingExperiment(
        name="sicap_weak_consistency",
        base="config/sicapv2_wsi_weak.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "image_sicap_paper_image_6_layer_no_MeanDatasetDice",
                            "image_sicap_paper_image_6_layer_no_MeanDatasetDice",
                            "image_sicap_paper_image_6_layer_no_MeanDatasetDice",
                            "image_sicap_paper_image_6_layer_no_MeanDatasetDice",
                            "image_sicap_paper_image_6_layer_no_loss",
                            "image_sicap_paper_image_6_layer_no_loss",
                            "image_sicap_paper_image_6_layer_no_loss",
                            "image_sicap_paper_image_6_layer_no_loss",
                            "image_sicap_paper_image_6_layer_no_GleasonScoreF1",
                            "image_sicap_paper_image_6_layer_no_GleasonScoreF1",
                            "image_sicap_paper_image_6_layer_no_GleasonScoreF1",
                            "image_sicap_paper_image_6_layer_no_GleasonScoreF1",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/652/abcfe75e46c34b208bf64433f91b3f4b/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/f76e089b612044879c02b1ed0aa1b845/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/d588a84f391344a69b25db67067c8621/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/5f23208a24a741d5a4c6f21c8917f32d/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/652/abcfe75e46c34b208bf64433f91b3f4b/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/f76e089b612044879c02b1ed0aa1b845/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/d588a84f391344a69b25db67067c8621/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/5f23208a24a741d5a4c6f21c8917f32d/artifacts/best.valid.graph.loss",
                        "s3://mlflow/652/abcfe75e46c34b208bf64433f91b3f4b/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                        "s3://mlflow/652/f76e089b612044879c02b1ed0aa1b845/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                        "s3://mlflow/652/d588a84f391344a69b25db67067c8621/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                        "s3://mlflow/652/5f23208a24a741d5a4c6f21c8917f32d/artifacts/best.valid.graph.segmentation.GleasonScoreF1",
                    ],
                ),
            ]
        ],
    )

    GNNTestingExperiment(
        name="sicap_strong_consistency",
        base="config/sicapv2_wsi_weak.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_sicap_paper_with_cat_MeanDatasetDice",
                            "tissue_sicap_paper_with_cat_MeanDatasetDice",
                            "tissue_sicap_paper_with_cat_MeanDatasetDice",
                            "tissue_sicap_paper_with_cat_MeanDatasetDice",
                            "tissue_sicap_paper_with_cat_loss",
                            "tissue_sicap_paper_with_cat_loss",
                            "tissue_sicap_paper_with_cat_loss",
                            "tissue_sicap_paper_with_cat_loss",
                            "tissue_sicap_paper_with_cat_GleasonScoreF1",
                            "tissue_sicap_paper_with_cat_GleasonScoreF1",
                            "tissue_sicap_paper_with_cat_GleasonScoreF1",
                            "tissue_sicap_paper_with_cat_GleasonScoreF1",
                            "tissue_sicap_paper_gnn_15",
                            "tissue_sicap_paper_gnn_15",
                            "tissue_sicap_paper_gnn_15",
                            "tissue_sicap_paper_gnn_15",
                            "tissue_sicap_paper_gnn_9",
                            "tissue_sicap_paper_gnn_9",
                            "tissue_sicap_paper_gnn_9",
                            "tissue_sicap_paper_gnn_9",
                            "tissue_sicap_paper_gnn_3",
                            "tissue_sicap_paper_gnn_3",
                            "tissue_sicap_paper_gnn_3",
                            "tissue_sicap_paper_gnn_3",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/650/52b37fac07be47ee892b14038a19a195/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/16a14cfd1d714a4a878d3c7717910ec3/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/ef3ccc730759485fb8ebe7bb92e64e68/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/75c8e04a8c7a41b28f642ad62fc5003f/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/52b37fac07be47ee892b14038a19a195/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/16a14cfd1d714a4a878d3c7717910ec3/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/ef3ccc730759485fb8ebe7bb92e64e68/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/75c8e04a8c7a41b28f642ad62fc5003f/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/52b37fac07be47ee892b14038a19a195/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/16a14cfd1d714a4a878d3c7717910ec3/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/ef3ccc730759485fb8ebe7bb92e64e68/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/75c8e04a8c7a41b28f642ad62fc5003f/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/cadc7bef58d54cb78c8c09e731e1db94/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/a33ef1bc7de54370a70b5d59d61e7df0/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/b7a02ad5f829483cbcd1513ff21ea987/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/995f8b30a1094d88923d0dbdbc1437a2/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/65f064060b7d455eb21d2a5224217d02/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/a48a18f9c4204768bb2b241e7b5b73ed/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/d4ac463d40c746139d3a7292b25abf35/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/5f1dd8b7fb6e450581ba4783a37d881b/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/e0509775f415452dab3bbe8c0d9cb699/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/b92f31d6334b407c9d552095657d4b76/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/4e5c31b858c442a38a0a60d70a9f765e/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/4d6eb42621d949a7b8182592121058bc/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                    ],
                ),
            ]
        ],
    )

    GNNTestingExperiment(
        name="sicap_partial_10",
        base="config/sicapv2_wsi_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_sicap_partial_10_MeanDatasetDice",
                            "tissue_sicap_partial_10_MeanDatasetDice",
                            "tissue_sicap_partial_10_MeanDatasetDice",
                            "tissue_sicap_partial_10_MeanDatasetDice",
                            "tissue_sicap_partial_10_loss",
                            "tissue_sicap_partial_10_loss",
                            "tissue_sicap_partial_10_loss",
                            "tissue_sicap_partial_10_loss",
                            "tissue_sicap_partial_10_GleasonScoreF1",
                            "tissue_sicap_partial_10_GleasonScoreF1",
                            "tissue_sicap_partial_10_GleasonScoreF1",
                            "tissue_sicap_partial_10_GleasonScoreF1",
                            "semi_sicap_partial_10_MeanDatasetDice",
                            "semi_sicap_partial_10_MeanDatasetDice",
                            "semi_sicap_partial_10_MeanDatasetDice",
                            "semi_sicap_partial_10_MeanDatasetDice",
                            "semi_sicap_partial_10_loss",
                            "semi_sicap_partial_10_loss",
                            "semi_sicap_partial_10_loss",
                            "semi_sicap_partial_10_loss",
                            "semi_sicap_partial_10_GleasonScoreF1",
                            "semi_sicap_partial_10_GleasonScoreF1",
                            "semi_sicap_partial_10_GleasonScoreF1",
                            "semi_sicap_partial_10_GleasonScoreF1",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/650/9001e8a5bbf9440ea6b9536c4b4061ac/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/43afe069e50243abb77ad2c71fd35961/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/23d569c6e6104b24ab5b6594a86122ba/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/b6c687b7e87b4294b70113a06d747ee4/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/650/9001e8a5bbf9440ea6b9536c4b4061ac/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/43afe069e50243abb77ad2c71fd35961/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/23d569c6e6104b24ab5b6594a86122ba/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/b6c687b7e87b4294b70113a06d747ee4/artifacts/best.valid.node.loss",
                        "s3://mlflow/650/9001e8a5bbf9440ea6b9536c4b4061ac/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/43afe069e50243abb77ad2c71fd35961/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/23d569c6e6104b24ab5b6594a86122ba/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/650/b6c687b7e87b4294b70113a06d747ee4/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/656/d8ad7c2ac5cf4c61a78ef83e25b2a0eb/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/86e10ff02cfe40b6818f34bbf6f5d9a6/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/f47ecbaf2849442d9fc13c93a8547e3a/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/7e18990a9ce846fb950af3ac900a2024/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/656/d8ad7c2ac5cf4c61a78ef83e25b2a0eb/artifacts/best.valid.node.loss",
                        "s3://mlflow/656/86e10ff02cfe40b6818f34bbf6f5d9a6/artifacts/best.valid.node.loss",
                        "s3://mlflow/656/f47ecbaf2849442d9fc13c93a8547e3a/artifacts/best.valid.node.loss",
                        "s3://mlflow/656/7e18990a9ce846fb950af3ac900a2024/artifacts/best.valid.node.loss",
                        "s3://mlflow/656/d8ad7c2ac5cf4c61a78ef83e25b2a0eb/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/656/86e10ff02cfe40b6818f34bbf6f5d9a6/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/656/f47ecbaf2849442d9fc13c93a8547e3a/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                        "s3://mlflow/656/7e18990a9ce846fb950af3ac900a2024/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                    ],
                ),
            ]
        ],
    )

    # ABLATIONS

    # GNN Layers
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_gnn_3",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "model", "gnn_config", "n_layers"], 3),
            Parameter(["train", "params", "nr_epochs"], 3000),

        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_gnn_2",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "model", "gnn_config", "n_layers"], 2),
            Parameter(["train", "params", "nr_epochs"], 3000),

        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="sicap_semi_normal",
        base="config/paper_sicap_semi.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "params", "nr_epochs"], 3000),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "params", "loss", "node_weight"], [0.5])],
    )

    # Handcrafted Features
    CPUPreprocessingExperiment(
        name="sicapv2_wsi_handcrafted",
        base="config/preprocessing_sicap_wsi_handcrafted.yml",
        queue="prod.long",
        cores=4,
        path=PATH,
    ).generate(
        fixed=[ 
            Parameter(
                ["params", "link_directory"],
                "v0_4000_low_handcrafted",
            )
        ]
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_handcrafted",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "data", "graph_directory"], "v0_4000_low_handcrafted"),
            Parameter(["train", "params", "nr_epochs"], 3000),
            Parameter(["train", "data", "use_augmentation_dataset"], False),
            Parameter(["train", "model", "gnn_config", "input_dim"], 65),
            Parameter(["train", "data", "normalize"], True)
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="sicap_paper_handcrafted",
        base="config/paper_sicap_semi.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "params", "nr_epochs"], 3000),
            Parameter(["train", "data", "graph_directory"], "v0_4000_low_handcrafted"),
            Parameter(["train", "data", "use_augmentation_dataset"], False),
            Parameter(["train", "model", "gnn_config", "input_dim"], 65),
            Parameter(["train", "data", "normalize"], True)
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "params", "loss", "node_weight"], [0.5])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_paper_handcrafted",
        base="config/paper_sicap_weak.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "graph_directory"], "v0_4000_low_handcrafted"),
            Parameter(["train", "data", "use_augmentation_dataset"], False),
            Parameter(["train", "model", "gnn_config", "input_dim"], 67),
            Parameter(["train", "data", "normalize"], True)
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    # Grid Graph
    CPUPreprocessingExperiment(
        name="sicapv2_wsi_grid",
        base="config/preprocessing_sicap_wsi.yml",
        queue="prod.long",
        cores=4,
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["superpixel", "params", "compactness"], 10000
            ),
            Parameter(
                ["params", "link_directory"],
                "v0_4000_low_grid",
            )
        ]
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_grid",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "data", "graph_directory"], "v0_4000_low_grid"),
            Parameter(["train", "params", "nr_epochs"], 3000),

        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="sicap_paper_grid",
        base="config/paper_sicap_semi.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "params", "nr_epochs"], 3000),
            Parameter(["train", "data", "graph_directory"], "v0_4000_low_grid"),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "params", "loss", "node_weight"], [0.5])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_paper_grid",
        base="config/paper_sicap_weak.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "graph_directory"], "v0_4000_low_grid"),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    # Grid no
    CPUPreprocessingExperiment(
        name="sicapv2_wsi_grid_no",
        base="config/preprocessing_sicap_wsi.yml",
        queue="prod.long",
        cores=4,
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["superpixel", "params", "compactness"], 10000
            ),
            Parameter(["superpixel", "params", "threshold"], 0.0),
            Parameter(
                ["params", "link_directory"],
                "v0_4000_no_grid",
            )
        ]
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_grid_no",
        base="config/paper_sicap_strong.yml",
        queue="prod.long",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "data", "graph_directory"], "v0_4000_no_grid"),
            Parameter(["train", "params", "nr_epochs"], 3000),

        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="sicap_paper_grid_no",
        base="config/paper_sicap_semi.yml",
        queue="prod.long",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "params", "nr_epochs"], 3000),
            Parameter(["train", "data", "graph_directory"], "v0_4000_no_grid"),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "params", "loss", "node_weight"], [0.5])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_paper_grid_no",
        base="config/paper_sicap_weak.yml",
        queue="prod.long",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "graph_directory"], "v0_4000_no_grid"),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    # ResNet
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_resnet",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "data", "graph_directory"], "v0_low_4000_resnet34"),
            Parameter(["train", "params", "nr_epochs"], 3000),
            Parameter(["train", "model", "gnn_config", "input_dim"], 512)

        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="sicap_paper_resnet",
        base="config/paper_sicap_semi.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "params", "nr_epochs"], 3000),
            Parameter(["train", "data", "graph_directory"], "v0_low_4000_resnet34"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 512)
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "params", "loss", "node_weight"], [0.5])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_paper_resnet",
        base="config/paper_sicap_weak.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "graph_directory"], "v0_low_4000_resnet34"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 514)
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    # No merging
    CPUPreprocessingExperiment(
        name="sicapv2_wsi_no",
        base="config/preprocessing_sicap_wsi.yml",
        queue="prod.long",
        cores=4,
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["superpixel", "params", "threshold"], 0.0
            ),
            Parameter(
                ["params", "link_directory"],
                "v0_4000_no",
            )
        ]
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_no",
        base="config/paper_sicap_strong.yml",
        queue="prod.long",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "data", "graph_directory"], "v0_4000_no"),
            Parameter(["train", "params", "nr_epochs"], 3000),

        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="sicap_paper_no",
        base="config/paper_sicap_semi.yml",
        queue="prod.long",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "data", "augmentation_mode"], "node"),
            Parameter(["train", "params", "nr_epochs"], 3000),
            Parameter(["train", "data", "graph_directory"], "v0_4000_no"),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "params", "loss", "node_weight"], [0.5])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_paper_no",
        base="config/paper_sicap_weak.yml",
        queue="prod.long",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "graph_directory"], "v0_4000_no"),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )

    # No GNN
    for DIM in [24, 32, 48, 64]:
        StronglySupervisedGraphClassificationExperiment(
            name=f"sicap_paper_no_gnn_{DIM}",
            base="config/paper_sicap_strong.yml",
            queue="prod.long",
            path=PATH,
        ).generate(
            fixed=[
                Parameter(
                    ["train", "params", "optimizer", "scheduler"],
                    {
                        "class": "ReduceLROnPlateau",
                        "params": {
                            "mode": "max",
                            "factor": 0.5,
                            "patience": 10,
                            "min_lr": 0.000005,
                        },
                    },
                ),
                Parameter(
                    ["train", "params", "optimizer", "params", "lr"],
                    0.0001,
                ),
                Parameter(["train", "model", "gnn_config", "n_layers"], 2),
                Parameter(["train", "model", "gnn_config", "output_dim"], DIM),
                Parameter(["train", "model", "gnn_config", "hidden_dim"], DIM),
                Parameter(["train", "model", "node_classifier_config", "hidden_dim"], DIM),
                Parameter(["train", "params", "nr_epochs"], 3000),

            ],
            grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        )
        SemiSupervisedGraphClassificationExperiment(
            name=f"sicap_paper_no_gnn_{DIM}",
            base="config/paper_sicap_semi.yml",
            queue="prod.long",
            path=PATH,
        ).generate(
            fixed=[
                Parameter(["train", "data", "centroid_features"], "no"),
                Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
                Parameter(["train", "params", "use_weighted_loss"], True),
                Parameter(["train", "data", "augmentation_mode"], "node"),
                Parameter(["train", "params", "nr_epochs"], 3000),
                Parameter(["train", "model", "gnn_config", "n_layers"], 2),
                Parameter(["train", "model", "gnn_config", "output_dim"], DIM),
                Parameter(["train", "model", "gnn_config", "hidden_dim"], DIM),
                Parameter(["train", "model", "node_classifier_config", "hidden_dim"], DIM),
                Parameter(["train", "model", "graph_classifier_config", "hidden_dim"], DIM),
            ],
            grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
            sequential=[ParameterList(["train", "params", "loss", "node_weight"], [0.5])],
        )
        WeaklySupervisedGraphClassificationExperiment(
            name=f"sicap_paper_no_gnn_{DIM}",
            base="config/paper_sicap_weak.yml",
            queue="prod.long",
            path=PATH,
        ).generate(
            fixed=[
                Parameter(["train", "data", "graph_directory"], "v0_4000_no"),
                Parameter(["train", "model", "gnn_config", "output_dim"], DIM),
                Parameter(["train", "model", "gnn_config", "hidden_dim"], DIM),
                Parameter(["train", "model", "graph_classifier_config", "hidden_dim"], DIM),
            ],
            grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        )

    for thres in [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
        GNNTestingExperiment(
            name=f"sicap_paper_thres_{thres}",
            base="config/sicapv2_wsi_strong.yml",
            path=PATH,
        ).generate(
            fixed=[
                Parameter(["test", "params", "threshold"], thres)
            ],
            sequential=[
                [
                    ParameterList(
                        ["test", "params", "experiment_tags"],
                        [
                            {"grid_search": x}
                            for x in [
                                f"tissue_sicap_paper_thres_{thres}",
                            ]*4
                        ],
                    ),
                    ParameterList(
                        ["test", "model", "architecture"],
                        [
                            f"s3://mlflow/650/{run_id}/artifacts/best.valid.node.segmentation.GleasonScoreF1"
                            for run_id in [
                            "e07d11ddc4f446e3a5ed3e4b0bb78d78",
                            "b7297c5d0d1e4d92a77f115f0e2d271a",
                            "27f250a920a9419a9bca303b3554e4be",
                            "5a7dd480a8fb48cf94868d50d03ea447"
                            ]
                        ],
                    ),
                ]
            ],
        )

    # Augmentations
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_graph_aug",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "params", "nr_epochs"], 3000),
            Parameter(["train", "data", "augmentation_mode"], "graph"),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="sicap_paper_graph_aug",
        base="config/paper_sicap_semi.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "data", "augmentation_mode"], "graph"),
            Parameter(["train", "params", "nr_epochs"], 3000),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "params", "loss", "node_weight"], [0.5])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_paper_graph_aug",
        base="config/paper_sicap_weak.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "graph")
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="sicap_paper_no_aug",
        base="config/paper_sicap_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "params", "optimizer", "scheduler"],
                {
                    "class": "ReduceLROnPlateau",
                    "params": {
                        "mode": "max",
                        "factor": 0.5,
                        "patience": 10,
                        "min_lr": 0.000005,
                    },
                },
            ),
            Parameter(
                ["train", "params", "optimizer", "params", "lr"],
                0.0001,
            ),
            Parameter(["train", "params", "nr_epochs"], 3000),
            Parameter(["train", "data", "augmentation_mode"], "no"),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="sicap_paper_no_aug",
        base="config/paper_sicap_semi.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "centroid_features"], "no"),
            Parameter(["train", "model", "gnn_config", "input_dim"], 1280),
            Parameter(["train", "params", "use_weighted_loss"], True),
            Parameter(["train", "data", "augmentation_mode"], "no"),
            Parameter(["train", "params", "nr_epochs"], 3000),
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
        sequential=[ParameterList(["train", "params", "loss", "node_weight"], [0.5])],
    )
    WeaklySupervisedGraphClassificationExperiment(
        name="sicap_paper_no_aug",
        base="config/paper_sicap_weak.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(["train", "data", "augmentation_mode"], "no")
        ],
        grid=[ParameterList(["train", "data", "fold"], [1, 2, 3, 4])],
    )
    