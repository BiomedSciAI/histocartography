#!/usr/bin/env python3
"""
Script for training graph-based histocartography models
"""
import logging
import sys
import argparse
import tempfile
import importlib

import torch
import mlflow
import pytorch_lightning as pl

from brontes import Brontes
from histocartography.utils.io import read_params
from histocartography.dataloader.pascale_dataloader import make_data_loader
from histocartography.ml.models.constants import AVAILABLE_MODEL_TYPES, MODEL_TYPE, MODEL_MODULE
from histocartography.evaluation.evaluator import AccuracyEvaluator, ConfusionMatrixEvaluator

# setup logging
log = logging.getLogger('Histocartography::Training')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
h1.setFormatter(formatter)
log.addHandler(h1)

# Cuda support
CUDA = torch.cuda.is_available()

# configure argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-d',
    '--data_path',
    type=str,
    help='path to the data.',
    default='data/',
    required=False
)
parser.add_argument(
    '-conf',
    '--config_fpath',
    type=str,
    help='path to the config file.',
    default='',
    required=False
)
parser.add_argument(
    '-p',
    '--number_of_workers',
    type=int,
    help='number of workers.',
    default=1,
    required=False
)
parser.add_argument(
    '-r',
    '--train_ratio',
    type=int,
    help='% of data to use for training.',
    default=0.8,
    required=False
)
parser.add_argument(
    '-n',
    '--model_name',
    type=str,
    help='model name.',
    default='model',
    required=False
)
parser.add_argument(
    '-b',
    '--batch_size',
    type=int,
    help='batch size.',
    default=8,
    required=False
)
parser.add_argument(
    '--epochs', type=int, help='epochs.', default=10, required=False
)
parser.add_argument(
    '-l',
    '--learning_rate',
    type=float,
    help='learning rate.',
    default=10e-3,
    required=False
)


def main(arguments):
    """
    Train SqueezeNet with brontes.
    Args:
        arguments (Namespace): parsed arguments.
    """
    # create aliases
    DATA_PATH = arguments.data_path
    NUMBER_OF_WORKERS = arguments.number_of_workers
    MODEL_NAME = arguments.model_name
    BATCH_SIZE = arguments.batch_size
    EPOCHS = arguments.epochs
    LEARNING_RATE = arguments.learning_rate
    TRAIN_RATIO = arguments.train_ratio

    # load config file
    config = read_params(arguments.config_fpath, verbose=True)

    # make data loaders (train & validation)
    dataloaders = make_data_loader(
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        num_workers=NUMBER_OF_WORKERS,
        path=DATA_PATH,
        config=config,
        cuda=CUDA
    )

    # declare model
    model_type = config[MODEL_TYPE]
    if model_type in list(AVAILABLE_MODEL_TYPES.keys()):
        module = importlib.import_module(
            MODEL_MODULE.format(model_type)
        )
        model = getattr(module, AVAILABLE_MODEL_TYPES[model_type])(config)
    else:
        raise ValueError(
            'Graph builder type: {} not recognized. Options are: {}'.format(
                model_type, list(AVAILABLE_MODEL_TYPES.keys())
            )
        )

    # build optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # mlflow log parameters
    mlflow.log_params({
        'number_of_workers': NUMBER_OF_WORKERS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'config': config
    })

    # define metrics
    accuracy_evaluation = AccuracyEvaluator(cuda=CUDA)
    confusion_matrix_evaluation = ConfusionMatrixEvaluator(cuda=CUDA)
    metrics = {
        'accuracy': accuracy_evaluation,
        'confusion_matrix': confusion_matrix_evaluation
    }

    # define brontes model
    brontes_model = Brontes(
        model=model,
        loss=loss_fn,
        data_loaders=dataloaders,
        optimizers=optimizer,
        training_log_interval=10,
        tracker_type='mlflow',
        metrics=metrics
    )

    # finally, train the model
    if CUDA:
        trainer = pl.Trainer(gpus=[0], max_nb_epochs=EPOCHS)
    else:
        trainer = pl.Trainer(max_nb_epochs=EPOCHS)
    
    trainer.fit(brontes_model)

    # save the model to tmp and log it as an mlflow artifact
    saved_model = f'{tempfile.mkdtemp()}/{MODEL_NAME}.pt'
    torch.save(brontes_model.model, saved_model)
    mlflow.log_artifact(saved_model)


if __name__ == "__main__":
    main(arguments=parser.parse_args())
