#!/usr/bin/env python3
"""
Script for training histocartography models
"""
import logging
import sys
import argparse
import tempfile
import os
import glob

import torch
import mlflow
import numpy as np
import pytorch_lightning as pl

from brontes import Brontes
from histocartography.ml.models import UNet
import histocartography.io.utils as utils
from histocartography.ml.datasets import WSIPatchSegmentationDataset

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::Training')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
h1.setFormatter(formatter)
log.addHandler(h1)

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
    '-t',
    '--dataset',
    type=str,
    help='pattern to match for the dataset in the s3 remote.',
    default='prostate/TMA/gleason2019',
    required=False
)
parser.add_argument(
    '--bucket',
    type=str,
    help='s3 bucket',
    default='test-data',
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
    '-n',
    '--model_name',
    type=str,
    help='model name.',
    default='model',
    required=False
)
parser.add_argument(
    '-s',
    '--seed',
    type=int,
    help='seed for reproducible results.',
    default=42,
    required=False
)
parser.add_argument(
    '-b',
    '--batch_size',
    type=int,
    help='batch size.',
    default=5,
    required=False
)
parser.add_argument(
    '--epochs', type=int, help='epochs.', default=2, required=False
)
parser.add_argument(
    '-l',
    '--learning_rate',
    type=float,
    help='learning rate.',
    default=1e-5,
    required=False
)

parser.add_argument(
    '--unet_depth',
    type=int,
    help='depth of UNet',
    default=5,
    required=False
)

parser.add_argument(
    '--filters',
    type=int,
    help='initial number of filters for UNet',
    default=16,
    required=False
)

parser.add_argument(
    '--patch_size',
    type=int,
    help='Patch Size',
    default=128,
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
    BUCKET = arguments.bucket
    DATASET = arguments.dataset
    NUMBER_OF_WORKERS = arguments.number_of_workers
    MODEL_NAME = arguments.model_name
    SEED = arguments.seed
    BATCH_SIZE = arguments.batch_size
    EPOCHS = arguments.epochs
    LEARNING_RATE = arguments.learning_rate
    UNET_DEPTH = arguments.unet_depth
    FILTERS = arguments.filters
    PATCH_SIZE = arguments.patch_size

    # make sure the data folder exists
    os.makedirs(DATA_PATH, exist_ok=True)

    # set the seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # data loaders for the GLEASON 2019 dataset
    utils.download_s3_dataset(
        utils.get_s3(), BUCKET, DATASET, DATA_PATH
    )
    # Get a list of all images
    all_img_files = glob.glob(
        os.path.join(DATA_PATH, DATASET, 'Train Imgs', '*.jpg')
    )
    label_image_pairs = {}
    for filename in all_img_files:
        slide, core = os.path.splitext(os.path.basename(filename)
                                       )[0].split('_')
        corresponding_img = f'{slide}_{core}.jpg'
        label_image_pairs[f'{slide}_{core}_classimg_nonconvex.png'
                          ] = os.path.join(
                              DATA_PATH, DATASET, 'Train Imgs',
                              corresponding_img
                          )

    # Choose a set of annotations
    annotation_subpath = 'Maps*'
    label_folder = np.random.choice(
        glob.glob(f'{os.path.join(DATA_PATH,DATASET, annotation_subpath)}')
    )
    label_folder_content = glob.glob(os.path.join(label_folder, '*.png'))

    # Find the names of the annotation files if
    #  label_image_pairs[os.path.basename(label_file)]
    pairs = [
        (label_file, label_image_pairs.get(os.path.basename(label_file)))
        for label_file in label_folder_content
        if os.path.basename(label_file) in label_image_pairs
    ]
    log.debug(pairs)


    full_dataset = torch.utils.data.ConcatDataset(
        [
            WSIPatchSegmentationDataset(image, label, (PATCH_SIZE, PATCH_SIZE), (PATCH_SIZE, PATCH_SIZE))
            for label, image in pairs
        ]
    )
    train_length = int(len(full_dataset) * 0.8)
    val_length = len(full_dataset) - train_length
    training_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_length, val_length]
    )

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
    number_of_filters = [FILTERS * pow(2,level) for level in range(DEPTH) ]
    base_model = UNet(number_of_filters=number_of_filters)

    optimizer = torch.optim.Adam(
        base_model.parameters(),
        lr=LEARNING_RATE
    )

    # mlflow set a tag
    mlflow.set_tag('label_folder', label_folder)
    # mlflow log parameters
    # mlflow.log_params({
    #     'number_of_workers': NUMBER_OF_WORKERS,
    #     'seed': SEED,
    #     'batch_size': BATCH_SIZE,
    #     'learning_rate': LEARNING_RATE
    # })

    # brontes model is initialized with base_model, optimizer, loss,
    # data_loaders. Optionally a dict of metrics functions and a
    # batch_fn applied to every batch can be provided.
    brontes_model = Brontes(
        model=base_model,
        loss=torch.nn.MSELoss(),
        data_loaders=dataset_loaders,
        optimizers=optimizer,
        training_log_interval=10,
        tracker_type='mlflow'
    )

    # finally, train the model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = pl.Trainer(gpus=[0], max_nb_epochs=EPOCHS)
    trainer.fit(brontes_model)

    # save the model to tmp and log it as an mlflow artifact
    saved_model = f'{tempfile.mkdtemp()}/{MODEL_NAME}.pt'
    torch.save(brontes_model.model, saved_model)
    mlflow.log_artifact(saved_model)


if __name__ == "__main__":
    main(arguments=parser.parse_args())
