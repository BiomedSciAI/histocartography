import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import GleasonPercentageDataset, GraphBatch, collate_graphs
from inference import GraphGradCAMBasedInference
from logging_helper import (
    log_confusion_matrix,
    log_device,
    log_nr_parameters,
    prepare_experiment,
    robust_mlflow,
)
from models import (
    ImageTissueClassifier,
    MLPModel,
    SemiSuperPixelTissueClassifier,
    SuperPixelTissueClassifier,
)
from utils import dynamic_import_from, get_batched_segmentation_maps, get_config

with os.popen("hostname") as subprocess:
    hostname = subprocess.read()
if hostname.startswith("zhcc"):
    SCRATCH_PATH = Path("/dataL/anv/")
    if not SCRATCH_PATH.exists():
        try:
            SCRATCH_PATH.mkdir()
        except FileExistsError:
            pass
else:
    SCRATCH_PATH = Path("/tmp")

MULTI_LABELS_TO_GG = {
    str([1, 0, 0, 0]): 0,  # benign
    str([0, 1, 0, 0]): 1,  # Grade6
    str([0, 1, 1, 0]): 2,  # Grade7
    str([0, 0, 1, 0]): 3,  # Grade8
    str([0, 1, 0, 1]): 3,  # Grade8
    str([0, 0, 1, 1]): 4,  # Grade9
    str([0, 0, 0, 1]): 5,  # Grade10
}

GG_SUM_TO_LABEL = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

CLASSES = ["Benign", "Grade6", "Grade7", "Grade8", "Grade9", "Grade10"]


def get_model(architecture):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    if architecture.startswith("s3://mlflow"):
        model = mlflow.pytorch.load_model(architecture, map_location=device)
    elif architecture.endswith(".pth"):
        model = torch.load(architecture, map_location=device)
    else:
        raise NotImplementedError(f"Cannot test on this architecture: {architecture}")
    return model


def parse_mlflow_list(x: str):
    return list(map(float, x[1:-1].split(",")))


def fill_missing_information(model_config, data_config):
    if model_config["architecture"].startswith("s3://mlflow"):
        _, _, _, experiment_id, run_id, _, _ = model_config["architecture"].split("/")
        df = mlflow.search_runs(experiment_id)
        df = df.set_index("run_id")

        if (
            "use_augmentation_dataset" not in data_config
            and "params.data.use_augmentation_dataset" in df
        ):
            data_config["use_augmentation_dataset"] = (
                df.loc[run_id, "params.data.use_augmentation_dataset"] == "True"
            )
        if "graph_directory" not in data_config:
            data_config["graph_directory"] = df.loc[
                run_id, "params.data.graph_directory"
            ]
        if "centroid_features" not in data_config:
            data_config["centroid_features"] = df.loc[
                run_id, "params.data.centroid_features"
            ]
        if (
            "normalize_features" not in data_config
            and "params.data.normalize_features" in df
        ):
            data_config["normalize_features"] = (
                df.loc[run_id, "params.data.normalize_features"] == "True"
            )
        if "fold" not in data_config and "params.data.fold" in df:
            data_config["fold"] = int(df.loc[run_id, "params.data.fold"])


def get_labels_df(LABELS_PATH):
    labels_df = pd.read_excel(LABELS_PATH, engine="openpyxl")

    def score_to_label(x):
        if x == 0:
            return x
        else:
            return x - 2

    labels_df["Gleason_primary"] = labels_df["Gleason_primary"].apply(score_to_label)
    labels_df["Gleason_secondary"] = labels_df["Gleason_secondary"].apply(
        score_to_label
    )
    labels_df = labels_df.set_index("slide_id")
    return labels_df


def assign_group(primary, secondary):
    def assign(a, b):
        if (a > 0) and (b == 0):
            b = a
        if (b > 0) and (a == 0):
            a = b
        return a, b

    if isinstance(primary, int) and isinstance(secondary, int):
        a, b = assign(primary, secondary)
        return GG_SUM_TO_LABEL[a + b]
    else:
        gg = []
        for a, b in zip(primary, secondary):
            a, b = assign(a, b)
            gg.append(GG_SUM_TO_LABEL[a + b])
        return np.array(gg)


def gleason_summary_wsum(y_pred, thres=None):
    gleason_scores = y_pred.copy()
    # gleason_scores /= np.sum(gleason_scores)
    # remove outlier predictions
    if thres is not None:
        gleason_scores[gleason_scores < thres] = 0
    # and assign overall grade
    idx = np.argsort(gleason_scores)[::-1]
    primary_class = int(idx[0])
    secondary_class = int(idx[1]) if gleason_scores[idx[1]] > 0 else int(idx[0])
    final_class = assign_group(primary_class, secondary_class)
    return final_class


def multihead_loss(out, labels, device):
    primary, secondary = out
    primary, secondary = primary.to(device), secondary.to(device)
    primary_labels, secondary_labels = labels
    primary_labels, secondary_labels = primary_labels.to(device), secondary_labels.to(
        device
    )
    if len(list(primary.shape)) == 1:
        primary = primary.unsqueeze(dim=0)
        secondary = secondary.unsqueeze(dim=0)
    loss = F.cross_entropy(primary, primary_labels) + F.cross_entropy(
        secondary, secondary_labels
    )
    return loss


def multilabel_loss(out, labels, device):
    out, labels = out.to(device), labels.to(device)
    if len(list(out.shape)) == 1:
        out = out.unsqueeze(dim=0)
    loss = F.binary_cross_entropy_with_logits(
        input=out, target=labels.to(torch.float32)
    )
    return loss


def finalgleasongrade_loss(out, labels):
    out, labels = out.to(device), labels.to(device)
    if len(list(out.shape)) == 1:
        out = out.unsqueeze(dim=0)
    loss = F.cross_entropy(out, labels)
    return loss


def build_loss_fn(mode, device):
    if mode == "multihead":
        return partial(multihead_loss, device=device)
    elif mode == "multilabel":
        return partial(multilabel_loss, device=device)
    elif mode == "finalgleasonscore":
        return partial(finalgleasongrade_loss, device=device)
    else:
        raise ValueError("Unsupported mode")


def create_dataset(
    dataset: str,
    data_config: Dict,
    model_config: Dict,
    batch_size: int = 8,
    num_workers: int = 8,
    test: bool = False,
    mode: str = "multihead",
    normalize_percentages: bool = False,
    **kwargs,
):
    NR_CLASSES = dynamic_import_from(dataset, "NR_CLASSES")
    VARIABLE_SIZE = dynamic_import_from(dataset, "VARIABLE_SIZE")
    LABELS_PATH = dynamic_import_from(dataset, "LABELS_PATH")
    prepare_graph_datasets = dynamic_import_from(dataset, "prepare_graph_datasets")
    prepare_graph_testset = dynamic_import_from(dataset, "prepare_graph_testset")

    train_arguments = data_config.copy()
    train_arguments.update(
        {
            "overfit_test": test,
            "additional_training_arguments": {
                "return_segmentation_info": True,
                "segmentation_downsample_ratio": 2,
            },
            "additional_validation_arguments": {"segmentation_downsample_ratio": 2},
        }
    )
    training_dataset, validation_dataset = prepare_graph_datasets(
        **train_arguments,
    )
    test_arguments = data_config.copy()
    test_arguments.update({"test": test, "segmentation_downsample_ratio": 2})
    test_dataset = prepare_graph_testset(**test_arguments)

    device = log_device()
    model = get_model(**model_config)
    model = model.to(device)
    model.eval()
    if isinstance(model, ImageTissueClassifier):
        mode = "weak_supervision"
    elif isinstance(model, SemiSuperPixelTissueClassifier):
        mode = "semi_strong_supervision"
    elif isinstance(model, SuperPixelTissueClassifier):
        mode = "strong_supervision"
    else:
        raise NotImplementedError

    training_loader = DataLoader(
        training_dataset,
        batch_size=1 if VARIABLE_SIZE else batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1 if VARIABLE_SIZE else batch_size,
        collate_fn=collate_graphs,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1 if VARIABLE_SIZE else batch_size,
        collate_fn=collate_graphs,
        num_workers=num_workers,
    )

    # Do forward pass of GNN model
    train_df, valid_df, test_df = extract_all(
        model=model,
        device=device,
        mode=mode,
        training_loader=training_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        NR_CLASSES=NR_CLASSES,
    )

    # Calculate percentage datasets
    labels_df = get_labels_df(LABELS_PATH)
    train_dataset = GleasonPercentageDataset(
        train_df, labels_df, mode="multihead", normalize=normalize_percentages
    )
    valid_dataset = GleasonPercentageDataset(
        valid_df,
        labels_df,
        mode="multihead",
        normalize=train_dataset if normalize_percentages else False,
    )
    test_dataset = GleasonPercentageDataset(
        test_df,
        labels_df,
        mode="multihead",
        normalize=train_dataset if normalize_percentages else False,
    )

    return train_dataset, valid_dataset, test_dataset


def extract_all(
    model, device, mode, training_loader, validation_loader, test_loader, NR_CLASSES
):
    run_id = robust_mlflow(mlflow.active_run).info.run_id
    train = run_forward_pass(
        model=model,
        device=device,
        mode=mode,
        loader=training_loader,
        NR_CLASSES=NR_CLASSES,
    )
    with tempfile.TemporaryDirectory(prefix=run_id) as temp_dir:
        train_path = Path(temp_dir) / "train.csv"
        train.to_csv(train_path)
        robust_mlflow(
            mlflow.log_artifact,
            str(train_path),
            artifact_path="areas",
        )
    valid = run_forward_pass(
        model=model,
        device=device,
        mode=mode,
        loader=validation_loader,
        NR_CLASSES=NR_CLASSES,
    )
    with tempfile.TemporaryDirectory(prefix=run_id) as temp_dir:
        valid_path = Path(temp_dir) / "valid.csv"
        valid.to_csv(valid_path)
        robust_mlflow(
            mlflow.log_artifact,
            str(valid_path),
            artifact_path="areas",
        )
    test = run_forward_pass(
        model=model, device=device, mode=mode, loader=test_loader, NR_CLASSES=NR_CLASSES
    )
    with tempfile.TemporaryDirectory(prefix=run_id) as temp_dir:
        test_path = Path(temp_dir) / "test.csv"
        test.to_csv(test_path)
        robust_mlflow(
            mlflow.log_artifact,
            str(test_path),
            artifact_path="areas",
        )
    return train, valid, test


def run_forward_pass(model, device, mode, loader, NR_CLASSES) -> pd.DataFrame:
    model.eval()
    percentages = list()
    graph_batch: GraphBatch
    for graph_batch in tqdm(loader, total=len(loader)):
        names = graph_batch.names
        graph = graph_batch.meta_graph.to(device)
        if mode == "weak_supervision":
            logits = model(graph)
            inferencer = GraphGradCAMBasedInference(NR_CLASSES, model, device=device)
            segmentation_maps = inferencer.predict_batch(
                graph_batch.meta_graph, graph_batch.instance_maps
            )
        else:
            with torch.no_grad():
                logits = model(graph)
                segmentation_maps = get_batched_segmentation_maps(
                    node_logits=logits,
                    node_associations=graph.batch_num_nodes,
                    superpixels=graph_batch.instance_maps,
                    NR_CLASSES=NR_CLASSES,
                )
        for segmentation_map, name in zip(segmentation_maps, names):
            values, counts = np.unique(segmentation_map, return_counts=True)
            all_counts = np.zeros(NR_CLASSES, dtype=np.int64)
            all_counts[values] = counts
            percentages.append((name,) + tuple(all_counts))
    classes = ["name"] + [f"class_{i}" for i in range(NR_CLASSES)]
    return pd.DataFrame(percentages, columns=classes).set_index("name")


def extract_gleason_grades(mode, logits, labels, loss_fn=None):
    if mode == "multihead":
        all_val_p_logits = torch.cat(list(zip(*logits))[0])
        all_val_p_labels = torch.cat(list(zip(*labels))[0])
        all_val_s_logits = torch.cat(list(zip(*logits))[1])
        all_val_s_labels = torch.cat(list(zip(*labels))[1])
        if loss_fn is not None:
            valid_loss = loss_fn(
                (all_val_p_logits, all_val_s_logits),
                (all_val_p_labels, all_val_s_labels),
            )
        pred_primary = torch.argmax(all_val_p_logits, dim=1).cpu().detach().numpy()
        pred_secondary = torch.argmax(all_val_s_logits, dim=1).cpu().detach().numpy()
        pred_gg = assign_group(pred_primary, pred_secondary)
        gg_labels = assign_group(
            all_val_p_labels.cpu().detach().numpy(),
            all_val_s_labels.cpu().detach().numpy(),
        )
    else:
        raise NotImplementedError

    if loss_fn is not None:
        return pred_gg, gg_labels, valid_loss
    else:
        return pred_gg, gg_labels


def train_mlp(
    training_dataset: GleasonPercentageDataset,
    validation_dataset: GleasonPercentageDataset,
    dataset: str,
    device,
    mode: str = "multihead",
    lr=0.0005,
    weight_decay=1e-5,
    batch_size=16,
    nr_epochs=10000,
    **kwargs,
):
    NR_CLASSES = dynamic_import_from(dataset, "NR_CLASSES")

    model = MLPModel(num_classes=NR_CLASSES, mode=mode)
    assert training_dataset.mode == mode
    assert validation_dataset.mode == mode
    log_nr_parameters(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    robust_mlflow(
        mlflow.log_params,
        {
            "lr": lr,
            "weight_decay": weight_decay,
            "mode": mode,
            "batch_size": batch_size,
        },
    )

    loss_fn = build_loss_fn(mode, device)

    train_loader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    f1_scores = list()
    kappa_scores = list()
    losses = list()
    best_epoch = 0
    best_loss = 1e5
    for epoch in tqdm(range(nr_epochs)):
        # A.) train for 1 epoch
        model.train()
        for inp, labels in train_loader:

            # 1. forward pass
            model_output = model(
                inp.to(device)
            )  # can be P+S, Final GG or Binary classification.

            # 2. backward pass
            loss = loss_fn(model_output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # B.) Evaluate
        model.eval()
        epoch_labels = list()
        epoch_logits = list()
        with torch.no_grad():
            for inp, labels in valid_loader:
                model_output = model(inp.to(device))
                epoch_logits.append(model_output)
                epoch_labels.append(labels)

        pred_gg, gg_labels, valid_loss = extract_gleason_grades(
            mode=mode, logits=epoch_logits, labels=epoch_labels, loss_fn=loss_fn
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            torch.save(model, "best_mlp.pt")

        weighted_f1_score = sklearn.metrics.f1_score(
            gg_labels, pred_gg, average="weighted"
        )
        kappa_score = sklearn.metrics.cohen_kappa_score(
            gg_labels, pred_gg, weights="quadratic"
        )
        f1_scores.append(weighted_f1_score)
        kappa_scores.append(kappa_score)
        losses.append(valid_loss.item())

    run_id = robust_mlflow(mlflow.active_run).info.run_id
    with tempfile.TemporaryDirectory(prefix=run_id) as tmpdir:
        sns.lineplot(y=losses, x=list(range(len(losses))))
        plt.plot([best_epoch, best_epoch], [min(losses), max(losses)])
        plt.title("Validation Loss")
        local_path = os.path.join(tmpdir, "valid_loss.png")
        plt.savefig(local_path, dpi=300, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(local_path)

        sns.lineplot(y=f1_scores, x=list(range(len(f1_scores))))
        plt.plot([best_epoch, best_epoch], [min(f1_scores), max(f1_scores)])
        plt.title("Validation F1Score")
        local_path = os.path.join(tmpdir, "valid_f1.png")
        plt.savefig(local_path, dpi=300, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(local_path)

        sns.lineplot(y=kappa_scores, x=list(range(len(kappa_scores))))
        plt.plot([best_epoch, best_epoch], [min(kappa_scores), max(kappa_scores)])
        plt.title("Validation KappaScore")
        local_path = os.path.join(tmpdir, "valid_kappa.png")
        plt.savefig(local_path, dpi=300, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(local_path)

        mlflow.log_artifact("best_mlp.pt")

    return model


def run_mlp(model, device, testing_dataset: GleasonPercentageDataset):
    test_loader = DataLoader(
        testing_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    mode = testing_dataset.mode

    model.eval()
    all_labels = list()
    all_logits = list()
    with torch.no_grad():
        for inp, labels in test_loader:
            model_output = model(inp.to(device))
            all_logits.append(model_output)
            all_labels.append(labels)

    pred_gg, gg_labels = extract_gleason_grades(
        mode=mode, logits=all_logits, labels=all_labels
    )

    weighted_f1_score = sklearn.metrics.f1_score(gg_labels, pred_gg, average="weighted")
    kappa_score = sklearn.metrics.cohen_kappa_score(
        gg_labels, pred_gg, weights="quadratic"
    )

    mlflow.log_metric("MLP.GleasonScoreF1", weighted_f1_score)
    mlflow.log_metric("MLP.GleasonScoreKappa", kappa_score)
    log_confusion_matrix(
        prediction=pred_gg, ground_truth=gg_labels, classes=CLASSES, name="Test.MLP"
    )


if __name__ == "__main__":
    config, config_path, test = get_config(
        name="test",
        default="config/default_strong.yml",
        required=("model", "data"),
    )
    fill_missing_information(config["model"], config["data"])
    prepare_experiment(config_path=config_path, **config)

    # Create percentage datasets
    training_dataset, validation_dataset, testing_dataset = create_dataset(
        model_config=config["model"],
        data_config=config["data"],
        test=test,
        **config["params"],
    )

    device = log_device()
    # Train MLP
    model = train_mlp(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        device=device,
        **config["params"],
    )

    # Evaluate on testset
    run_mlp(model=model, device=device, testing_dataset=testing_dataset)
