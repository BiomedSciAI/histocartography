#!/usr/bin/env python3
"""
Script for generating explanations, ie subgraph that are "explaning"
the prediction.

Implementation derived from the GNN-Explainer (NeurIPS'19)
"""

import importlib
import torch
import mlflow
from mlflow.pytorch import load_model

from histocartography.utils.io import read_params, unsafe_load_checkpoint
from histocartography.dataloader.pascale_dataloader import make_data_loader
from histocartography.ml.models.constants import AVAILABLE_MODEL_TYPES, MODEL_TYPE, MODEL_MODULE
from histocartography.utils.arg_parser import parse_arguments
from histocartography.ml.models.constants import load_superpx_graph, load_cell_graph
from histocartography.utils.io import get_device, flatten_dict
from histocartography.interpretability.single_instance_explainer import SingleInstanceExplainer
from histocartography.utils.graph import adj_to_networkx

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

    # make data loaders
    dataloaders, num_cell_features = make_data_loader(
        batch_size=1,
        num_workers=args.number_of_workers,
        path=args.data_path,
        config=config,
        cuda=CUDA,
        load_cell_graph=load_cell_graph(config['model_type']),
        load_superpx_graph=load_superpx_graph(config['model_type']),
        load_image=False
    )

    # define GNN model
    model_type = config[MODEL_TYPE]
    if model_type in list(AVAILABLE_MODEL_TYPES.keys()):
        module = importlib.import_module(
            MODEL_MODULE.format(model_type)
        )
        model = getattr(module, AVAILABLE_MODEL_TYPES[model_type])(
            config['model_params'], num_cell_features).to(DEVICE)
        if args.pretrained_model:
            fname = 's3://mlflow/fd2d1f71b3974e9a9afdab4112857a80/artifacts/trained_model/304279a8-e0a8-4a08-99b0-7ebc6264f48a'
            mlflow_model = load_model(fname)

            def is_int(s):
                try:
                    int(s)
                    return True
                except:
                    return False

            for n, p in mlflow_model.named_parameters():
                split = n.split('.')
                to_eval = 'model'
                for i, s in enumerate(split):
                    if is_int(s):
                        to_eval += '[' + s + ']'
                    else:
                        to_eval += '.'
                        to_eval += s
                print('To execute:', to_eval)
                exec(to_eval + '=' + 'p')

    else:
        raise ValueError(
            'Model: {} not recognized. Options are: {}'.format(
                model_type, list(AVAILABLE_MODEL_TYPES.keys())
            )
        )

    # mlflow log parameters
    inter_config = flatten_dict(config['explainer'])
    for key, val in inter_config.items():
        mlflow.log_param(key, val)

    # agg training parameters
    train_params = {
        'num_epochs': args.epochs,
        'lr': args.learning_rate,
        'weight_decay': 5e-4,
        'opt_decay_step': 0.1,
        'opt_decay_rate': 0.1
    }
    
    # declare explainer 
    explainer = SingleInstanceExplainer(
        model=model,
        train_params=train_params,
        model_params=config['explainer']
    )

    # explain instance from the train set
    for data, label in dataloaders['train']:

        cell_graph = data[0]

        adj, feats = explainer.explain(
            data=data,
            label=label
        )

        explanation = adj_to_networkx(adj, feats, rm_iso_nodes=True)

        print('Explanation:')
        print('Original Graph: # nodes: {} # edges: {}'.format(
            cell_graph.number_of_nodes(),
            cell_graph.number_of_edges()
        ))

        print('Explanation Graph: # nodes: {} # edges: {}'.format(
            explanation.number_of_nodes(),
            explanation.number_of_edges()
        ))


if __name__ == "__main__":
    main(args=parse_arguments())
