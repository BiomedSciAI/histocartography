#!/usr/bin/env python3
"""
Script for training histocartography models
"""
import logging
import sys
import argparse
import tempfile
import os

import torch
import mlflow
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pytorch_lightning as pl

from brontes import Brontes
from histocartography.ml.models import UNet

# logging setup
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('mlflow-enabled-mnist-training')

# configure argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--data_path', type=str,
    help='path to the data.', default='data/',
    required=False
)
parser.add_argument(
    '-p', '--number_of_workers', type=int,
    help='number of workers.', default=1,
    required=False
)
parser.add_argument(
    '-n', '--model_name', type=str,
    help='model name.', default='model',
    required=False
)
parser.add_argument(
    '-s', '--seed', type=int,
    help='seed for reproducible results.', default=42,
    required=False
)
parser.add_argument(
    '-b', '--batch_size', type=int,
    help='batch size.', default=25,
    required=False
)
parser.add_argument(
    '--epochs', type=int,
    help='epochs.', default=2,
    required=False
)
parser.add_argument(
    '-l', '--learning_rate', type=float,
    help='learning rate.', default=1e-5,
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
    SEED = arguments.seed
    BATCH_SIZE = arguments.batch_size
    EPOCHS = arguments.epochs
    LEARNING_RATE = arguments.learning_rate

    # make sure the data folder exists
    os.makedirs(DATA_PATH, exist_ok=True)

    # set the seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # data loaders for the MNIST dataset
    dataset_loaders = {
        'train':
            torch.utils.data.DataLoader(
                training_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUMBER_OF_WORKERS
            ),
        'val':
            torch.utils.data.DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUMBER_OF_WORKERS
            )
    }

    # definition of base model
    base_model = Net()

    optimizer = torch.optim.Adam(
        base_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5  # standard value
    )

    # mlflow set a tag
    mlflow.set_tag('model_name', MODEL_NAME)
    # mlflow log parameters
    mlflow.log_params({
        'number_of_workers': NUMBER_OF_WORKERS,
        'seed': SEED,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    })

    # brontes model is initialized with base_model, optimizer, loss,
    # data_loaders. Optionally a dict of metrics functions and a
    # batch_fn applied to every batch can be provided.
    brontes_model = Brontes(
        model=base_model,
        loss=torch.nn.NLLLoss(),
        data_loaders=dataset_loaders,
        optimizers=optimizer,
        training_log_interval=10,
        tracker_type='mlflow'
    )

    # finally, train the model
    trainer = pl.Trainer(max_nb_epochs=EPOCHS)
    trainer.fit(brontes_model)

    # save the model to tmp and log it as an mlflow artifact
    saved_model = f'{tempfile.mkdtemp()}/{MODEL_NAME}.pt'
    torch.save(brontes_model.model, saved_model)
    mlflow.log_artifact(saved_model)


if __name__ == "__main__":
    main(arguments=parser.parse_args())