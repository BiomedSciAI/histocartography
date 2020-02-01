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

from histocartography.utils.io import read_params
from histocartography.dataloader.pascale_dataloader import make_data_loader
from histocartography.ml.models.constants import AVAILABLE_MODEL_TYPES, MODEL_TYPE, MODEL_MODULE
from histocartography.evaluation.evaluator import AccuracyEvaluator, WeightedF1
from histocartography.utils.arg_parser import parse_arguments
from histocartography.ml.models.constants import load_superpx_graph, load_cell_graph
from histocartography.utils.io import get_device, get_filename, check_for_dir, complete_path, save_image
#from histocartography.utils.visualization import GraphVisualization


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

    # make data loaders (train & validation)
    dataloaders, num_cell_features = make_data_loader(
        batch_size=args.batch_size,
        num_workers=args.number_of_workers,
        path=args.data_path,
        config=config,
        cuda=CUDA,
        load_cell_graph=load_cell_graph(config['model_type']),
        load_superpx_graph=load_superpx_graph(config['model_type']),
        load_image=True
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
        lr=args.learning_rate
    )

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # mlflow log parameters
    mlflow.log_params({
        'number_of_workers': args.number_of_workers,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'graph_building': config['graph_building'],
        'gnn_params': config['model_params']['gnn_params'],
        'readout_params': config['model_params']['readout'],
        'model_type': config['model_type']
    })

    # define metrics
    accuracy_evaluation = AccuracyEvaluator(cuda=CUDA)
    weighted_f1_score = WeightedF1(cuda=CUDA)
    metrics = {
        'accuracy': accuracy_evaluation,
        'weighted_f1_score' : weighted_f1_score
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

    # train the model with pytorch lightning
    if CUDA:
        trainer = pl.Trainer(gpus=[0], max_nb_epochs=args.epochs)
    else:
        trainer = pl.Trainer(max_nb_epochs=args.epochs)

    trainer.fit(brontes_model)

    # Evaluation
    trainer.test(brontes_model)

    # save the model to tmp and log it as an mlflow artifact
    saved_model = f'{tempfile.mkdtemp()}/{args.model_name}.pt'
    torch.save(brontes_model.model, saved_model)
    mlflow.log_artifact(saved_model)
    # mlflow.pytorch.log_model(brontes_model.model, “artifacts/model”, conda_env=“conda.yml”)



if __name__ == "__main__":
    main(args=parse_arguments())
