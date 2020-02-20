#!/usr/bin/env python3
"""
Script for training graph-based histocartography models
"""
import logging
import sys
import tempfile
import importlib
import torch
import mlflow
import dgl
import os
import uuid
from PIL import Image
from tqdm import tqdm
import mlflow.pytorch
import pandas as pd
import numpy as np
import shutil

from histocartography.utils.io import read_params, check_for_dir
from histocartography.dataloader.pascale_dataloader import make_data_loader
from histocartography.ml.models.constants import AVAILABLE_MODEL_TYPES, MODEL_TYPE, MODEL_MODULE
from histocartography.evaluation.evaluator import AccuracyEvaluator, WeightedF1
from histocartography.evaluation.confusion_matrix import ConfusionMatrix
from histocartography.evaluation.classification_report import ClassificationReport
from histocartography.utils.arg_parser import parse_arguments
from histocartography.ml.models.constants import load_superpx_graph, load_cell_graph
from histocartography.utils.io import (
    get_device, get_filename, check_for_dir,
    complete_path, save_image, load_checkpoint,
    save_checkpoint, write_json, save_image
)

import warnings
warnings.filterwarnings("ignore")

# setup logging
log = logging.getLogger('Histocartography::Training')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
h1.setFormatter(formatter)
log.addHandler(h1)

# cuda support
CUDA = torch.cuda.is_available()
DEVICE = get_device(CUDA)


DATATYPE_TO_SAVEFN = {
    dict: write_json,
    np.ndarray: np.savetxt,
    Image.Image: save_image
}

DATATYPE_TO_EXT = {
    dict: '.json',
    np.ndarray: '.txt',
    Image.Image: '.png'
}


def main(args):
    """
    Train HistoGraph.
    Args:
        args (Namespace): parsed arguments.
    """

    # load config file
    config = read_params(args.config_fpath, verbose=True)

    # set model path
    model_path = complete_path(args.model_path, str(uuid.uuid4()))
    check_for_dir(model_path)

    # make data loaders (train & validation)
    dataloaders, num_cell_features = make_data_loader(
        batch_size=args.batch_size,
        num_workers=args.number_of_workers,
        path=args.data_path,
        config=config,
        cuda=CUDA,
        load_cell_graph=load_cell_graph(config['model_type']),
        load_superpx_graph=load_superpx_graph(config['model_type']),
        load_image=False
    )

    # declare model
    model_type = config[MODEL_TYPE]
    if model_type in list(AVAILABLE_MODEL_TYPES.keys()):
        module = importlib.import_module(
            MODEL_MODULE.format(model_type)
        )
        model = getattr(module, AVAILABLE_MODEL_TYPES[model_type])(
            config['model_params'], num_cell_features).to(DEVICE)
    else:
        raise ValueError(
            'Model: {} not recognized. Options are: {}'.format(
                model_type, list(AVAILABLE_MODEL_TYPES.keys())
            )
        )

    # build optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=5e-4
    )

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # mlflow log parameters
    mlflow.log_params({
        'number_of_workers': args.number_of_workers,
        'batch_size': args.batch_size
    })

    df = pd.io.json.json_normalize(config)
    rep = {"graph_building.": "", "model_params.": "", "gnn_params.": ""}  # replacement for shorter key names
    for i, j in rep.items():
        df.columns = df.columns.str.replace(i, j)
    flatten_config = df.to_dict(orient='records')[0]
    for key, val in flatten_config.items():
        mlflow.log_params({key: str(val)})

    # define metrics
    accuracy_evaluation = AccuracyEvaluator(cuda=CUDA)
    weighted_f1_score = WeightedF1(cuda=CUDA)
    conf_matrix = ConfusionMatrix(return_img=True)
    class_report = ClassificationReport()
    metrics = {
        'accuracy': accuracy_evaluation,
        'weighted_f1_score': weighted_f1_score,
    }
    evaluators = {
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

    # training loop
    step = 0
    best_val_loss = 10e5
    best_val_accuracy = 0.
    best_val_weighted_f1_score = 0.

    for epoch in range(args.epochs):
        # A.) train for 1 epoch
        for data, labels in tqdm(dataloaders['train'], desc='Epoch training {}'.format(epoch), unit='batch'):

            # 1. forward pass
            logits = model(data)

            # 2. backward pass
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 3. compute & store metrics
            mlflow.log_metric('loss', loss.item(), step=step)
            for m_name, m_fn in metrics.items():
                out = m_fn(logits, labels)
                mlflow.log_metric(m_name, out.item(), step=step)

            # 4. increment step
            step += 1

        # B.) validate
        all_val_logits = []
        all_val_labels = []
        for data, labels in tqdm(dataloaders['val'], desc='Epoch validation {}'.format(epoch), unit='batch'):
            logits = model(data)
            all_val_logits.append(logits)
            all_val_labels.append(labels)

        all_val_logits = torch.cat(all_val_logits)
        all_val_labels = torch.cat(all_val_labels)

        # compute & store loss + model
        loss = loss_fn(all_val_logits, all_val_labels).item()
        mlflow.log_metric('val_loss', loss, step=step)
        if loss < best_val_loss:
            best_val_loss = loss
            save_checkpoint(model, complete_path(model_path, 'model_best_val_loss.pt'))

        # compute & store accuracy + model
        accuracy = metrics['accuracy'](all_val_logits, all_val_labels).item()
        mlflow.log_metric('val_accuracy', accuracy, step=step)
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            save_checkpoint(model, complete_path(model_path, 'model_best_val_accuracy.pt'))

        # compute & store weighted f1-score + model
        weighted_f1_score = metrics['weighted_f1_score'](all_val_logits, all_val_labels).item()
        mlflow.log_metric('val_weighted_f1_score', weighted_f1_score, step=step)
        if weighted_f1_score > best_val_weighted_f1_score:
            best_val_weighted_f1_score = weighted_f1_score
            save_checkpoint(model, complete_path(model_path, 'model_best_val_weighted_f1_score.pt'))

    # testing loop
    for metric in ['best_val_loss', 'best_val_accuracy', 'best_val_weighted_f1_score']:

        model_name = [file for file in os.listdir(model_path) if file.endswith(".pt") and metric in file][0]
        load_checkpoint(model, complete_path(model_path, model_name))

        all_test_logits = []
        all_test_labels = []
        for data, labels in tqdm(dataloaders['test'], desc='Testing: {}'.format(metric), unit='batch'):
            logits = model(data)
            all_test_logits.append(logits)
            all_test_labels.append(labels)

        all_test_logits = torch.cat(all_test_logits)
        all_test_labels = torch.cat(all_test_labels)

        # compute & store loss
        loss = loss_fn(all_test_logits, all_test_labels).item()
        mlflow.log_metric('test_loss_' + metric, loss, step=step)

        # compute & store accuracy
        accuracy = metrics['accuracy'](all_test_logits, all_test_labels).item()
        mlflow.log_metric('test_accuracy_' + metric, accuracy, step=step)

        # compute & store weighted f1-score
        weighted_f1_score = metrics['weighted_f1_score'](all_test_logits, all_test_labels).item()
        mlflow.log_metric('test_weighted_f1_score_' + metric, weighted_f1_score, step=step)

        # run external evaluators
        for eval_name, eval_fn in evaluators.items():

            out = eval_fn(all_test_logits, all_test_labels)

            out_path = complete_path(model_path, eval_name)
            out_path += DATATYPE_TO_EXT[type(out)]

            DATATYPE_TO_SAVEFN[type(out)](out_path, out)

            artifact_path = 'evaluators/{}_{}'.format(eval_name, metric)
            mlflow.log_artifact(out_path, artifact_path=artifact_path)

        # log MLflow models
        mlflow.pytorch.log_model(model, 'model_' + metric)

    shutil.rmtree(model_path)


if __name__ == "__main__":
    main(args=parse_arguments())
