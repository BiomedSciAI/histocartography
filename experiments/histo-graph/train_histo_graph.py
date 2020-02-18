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
import pytorch_lightning as pl
from brontes import Brontes
import dgl
import os
import uuid

from histocartography.utils.io import read_params, check_for_dir
from histocartography.dataloader.pascale_dataloader import make_data_loader
from histocartography.ml.models.constants import AVAILABLE_MODEL_TYPES, MODEL_TYPE, MODEL_MODULE
from histocartography.evaluation.evaluator import AccuracyEvaluator, WeightedF1
from histocartography.evaluation.confusion_matrix import ConfusionMatrix
from histocartography.evaluation.classification_report import ClassificationReport
from histocartography.utils.arg_parser import parse_arguments
from histocartography.ml.models.constants import load_superpx_graph, load_cell_graph
from histocartography.utils.io import get_device, get_filename, check_for_dir, complete_path, save_image
import mlflow.pytorch
import pandas as pd

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
        weight_decay= 5e-4
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
    evaluator = {
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

    # define brontes model
    brontes_model = Brontes(
        model=model,
        loss=loss_fn,
        data_loaders=dataloaders,
        optimizers=optimizer,
        training_log_interval=10,
        tracker_type='mlflow',
        metrics=metrics,
        evaluators=evaluator,
        model_path=model_path,
    )

    # train the model with pytorch lightning
    early_stop = pl.callbacks.EarlyStopping('avg_val_loss', patience=40, mode='min')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='avg_val_loss'
    )
    if CUDA:
        trainer = pl.Trainer(
            gpus=[0],
            max_epochs=args.epochs,
            early_stop_callback=early_stop,
            checkpoint_callback=checkpoint_callback
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            early_stop_callback=early_stop,
            checkpoint_callback=checkpoint_callback
        )

    trainer.fit(brontes_model)
    
    # restore best model and test
    model_name = [file for file in os.listdir(model_path) if file.endswith(".ckpt")][0]
    brontes_model.load_state_dict(
        torch.load(
            os.path.join(model_path, model_name)
        )['state_dict']
    )
    trainer.test(brontes_model)


if __name__ == "__main__":
    main(args=parse_arguments())
