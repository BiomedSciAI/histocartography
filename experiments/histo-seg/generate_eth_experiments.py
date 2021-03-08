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

    folds = [
        ParameterList(
            ["train", "data", "training_slides"],
            [[111, 199, 204], [76, 111, 199], [204, 76, 111], [199, 204, 76]],
        ),
        ParameterList(
            ["train", "data", "validation_slides"], [[76], [204], [199], [111]]
        ),
    ]

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
                        {"grid_search": x + "_node"}
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

    GNNTestingExperiment(
        name="eth_rerun_tissue_partial_old",
        base="config/paper_eth_strong.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": "old_" + x}
                        for x in [
                            "tissue_eth_partial_5",
                            "tissue_eth_partial_5",
                            "tissue_eth_partial_5",
                            "tissue_eth_partial_5",
                            "tissue_eth_partial_10",
                            "tissue_eth_partial_10",
                            "tissue_eth_partial_10",
                            "tissue_eth_partial_10",
                            "tissue_eth_partial_25",
                            "tissue_eth_partial_25",
                            "tissue_eth_partial_25",
                            "tissue_eth_partial_25",
                            "tissue_eth_partial_50",
                            "tissue_eth_partial_50",
                            "tissue_eth_partial_50",
                            "tissue_eth_partial_50",
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
                        "s3://mlflow/645/ccdf24053d8f43e39096abe37a638cc2/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/499cfa735e724456997a7a7d78853d17/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/f5baa18334c34cf3a0c0b895b76191be/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/1034cb21c8a343fa979d1d883edb3ac8/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/507bb068fded4b869173870be7486468/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/89554ad5c09e46cab87878c51b50a6e1/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/8d5a01629de04d48a6e65029f3f70128/artifacts/best.valid.node.segmentation.fF1Score",
                        "s3://mlflow/645/71f342f710654121ae75433cdcbd41b7/artifacts/best.valid.node.segmentation.fF1Score",
                    ],
                ),
            ]
        ],
    )

    CPUPreprocessingExperiment(
        name="eth_partial_new", base="config/new_preprocess.yml", path=PATH
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
                    ["graph_builders", "params", "partial_annotation"],
                    [50, 25, 10, 5, 1],
                ),
                ParameterList(["params", "partial_annotation"], [50, 25, 10, 5, 1]),
                ParameterList(
                    ["params", "link_directory"],
                    [
                        f"v11_mobilenet_med_13x_partial_new_{s}"
                        for s in [50, 25, 10, 5, 1]
                    ],
                ),
            ]
        ],
    )

    SemiSupervisedGraphClassificationExperiment(
        name="eth_new_partial_100", base="config/paper_eth_semi.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x",
            ),
        ],
        sequential=[folds],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="eth_new_partial_50", base="config/paper_eth_semi.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_new_50",
            ),
            Parameter(["train", "data", "partial_annotation"], 50),
        ],
        sequential=[folds],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="eth_new_partial_25", base="config/paper_eth_semi.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_new_25",
            ),
            Parameter(["train", "data", "partial_annotation"], 25),
        ],
        sequential=[folds],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="eth_new_partial_10", base="config/paper_eth_semi.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_new_10",
            ),
            Parameter(["train", "data", "partial_annotation"], 10),
        ],
        sequential=[folds],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="eth_new_partial_5", base="config/paper_eth_semi.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_new_5",
            ),
            Parameter(["train", "data", "partial_annotation"], 5),
        ],
        sequential=[folds],
    )
    SemiSupervisedGraphClassificationExperiment(
        name="eth_new_partial_1", base="config/paper_eth_semi.yml", path=PATH
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_new_1",
            ),
            Parameter(["train", "data", "partial_annotation"], 1),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="eth_new_partial_100",
        base="config/paper_eth_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x",
            ),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="eth_new_partial_50",
        base="config/paper_eth_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_new_50",
            ),
            Parameter(["train", "data", "partial_annotation"], 50),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="eth_new_partial_25",
        base="config/paper_eth_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_new_25",
            ),
            Parameter(["train", "data", "partial_annotation"], 25),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="eth_new_partial_10",
        base="config/paper_eth_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_new_10",
            ),
            Parameter(["train", "data", "partial_annotation"], 10),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="eth_new_partial_5",
        base="config/paper_eth_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_new_5",
            ),
            Parameter(["train", "data", "partial_annotation"], 5),
        ],
        sequential=[folds],
    )
    StronglySupervisedGraphClassificationExperiment(
        name="eth_new_partial_1",
        base="config/paper_eth_strong.yml",
        path=PATH,
    ).generate(
        fixed=[
            Parameter(
                ["train", "data", "graph_directory"],
                "outputs/v11_mobilenet_med_13x_partial_new_1",
            ),
            Parameter(["train", "data", "partial_annotation"], 1),
        ],
        sequential=[folds],
    )

    GNNTestingExperiment(
        name="eth_rerun_semi_node",
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
                            "semi_eth_new_partial_100_node",
                            "semi_eth_new_partial_100_node",
                            "semi_eth_new_partial_100_node",
                            "semi_eth_new_partial_100_node",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/646/424978405a5a4d09baf871034bc0f4dc/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/646/c190b159066a4363acdf8d79334b49a5/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/646/f61f69569bcb4e6db5219113977b6f48/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                        "s3://mlflow/646/d612d349e39e4577be9a167333cf61b5/artifacts/best.valid.graph.segmentation.MeanDatasetDice",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="eth_rerun_semi_graph",
        base="config/paper_eth_weak.yml",
        path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_eth_new_partial_100_graph",
                            "semi_eth_new_partial_100_graph",
                            "semi_eth_new_partial_100_graph",
                            "semi_eth_new_partial_100_graph",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/646/424978405a5a4d09baf871034bc0f4dc/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/646/c190b159066a4363acdf8d79334b49a5/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/646/f61f69569bcb4e6db5219113977b6f48/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                        "s3://mlflow/646/d612d349e39e4577be9a167333cf61b5/artifacts/best.valid.node.segmentation.MeanDatasetDice",
                    ],
                ),
            ]
        ],
    )

    GNNTestingExperiment(
        name="eth_best_node_level", base="config/paper_eth_strong.yml", path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_rep_lr_normal_final",
                            "tissue_rep_lr_normal_final",
                            "tissue_rep_lr_normal_final",
                            "tissue_rep_lr_normal_final",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/645/5339c885ed5244d68efb89dc0882bb28/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/4a8d021f72d84695ac67a7895bd32992/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/89e0b32c52ba44ec93084fbb20b634ca/artifacts/best.valid.node.NodeClassificationF1Score",
                        "s3://mlflow/645/e8725f643d784f6688c6e1a503d2099d/artifacts/best.valid.node.NodeClassificationF1Score",
                    ],
                ),
            ]
        ],
    )

    GNNTestingExperiment(
        name="eth_tissue_stuck", base="config/paper_eth_strong.yml", path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "tissue_eth_new_partial_5",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/645/b07ea655d596489ab4e161561df26976/artifacts/best.valid.node.loss",
                    ],
                ),
            ]
        ],
    )
    GNNTestingExperiment(
        name="eth_semi_stuck", base="config/paper_eth_strong.yml", path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_eth_new_partial_25",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/646/a9a81e810c54404d9a60714e5db443c2/artifacts/best.valid.node.segmentation.GleasonScoreKappa",
                    ],
                ),
            ]
        ],
    )

    GNNTestingExperiment(
        name="eth_partial_25_consistent", base="config/paper_eth_strong.yml", path=PATH,
    ).generate(
        sequential=[
            [
                ParameterList(
                    ["test", "params", "experiment_tags"],
                    [
                        {"grid_search": x}
                        for x in [
                            "semi_eth_new_partial_25",
                        ]
                    ],
                ),
                ParameterList(
                    ["test", "model", "architecture"],
                    [
                        "s3://mlflow/646/64bf4cd651df4489ba2f3413f4d41f05/artifacts/best.valid.node.segmentation.GleasonScoreF1",
                    ],
                ),
            ]
        ],
    )
