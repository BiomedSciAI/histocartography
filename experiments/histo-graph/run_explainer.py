#!/usr/bin/env python3
"""
Script for generating explanations, ie subgraph that are "explaning"
the prediction.

Implementation derived from the GNN-Explainer (NeurIPS'19)
"""

import importlib
import torch
import mlflow
import dgl 
from mlflow.pytorch import load_model

from histocartography.utils.io import read_params
from histocartography.dataloader.pascale_dataloader import make_data_loader
from histocartography.ml.models.constants import AVAILABLE_MODEL_TYPES, MODEL_TYPE, MODEL_MODULE
from histocartography.utils.arg_parser import parse_arguments
from histocartography.ml.models.constants import load_superpx_graph, load_cell_graph
from histocartography.utils.io import get_device, flatten_dict
from histocartography.interpretability.single_instance_explainer import SingleInstanceExplainer
from histocartography.utils.graph import adj_to_networkx
from histocartography.utils.visualization import GraphVisualization

# flush warnings 
import warnings
warnings.filterwarnings("ignore")

# cuda support
CUDA = torch.cuda.is_available()
DEVICE = get_device(CUDA)

BASE_S3 = 's3://mlflow/'


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
        load_image=True,
        fold_id=3
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
            fname = BASE_S3 + 'd504d8ba7e7848098c7562a72e98e7bd/artifacts/model_best_val_weighted_f1_score_3'
            mlflow_model = load_model(fname,  map_location=torch.device('cpu'))

            def is_int(s):
                try:
                    int(s)
                    return True
                except:
                    return False

            for n, p in mlflow_model.named_parameters():
                split = n.split('.')
                to_eval = 'model'
                for s in split:
                    if is_int(s):
                        to_eval += '[' + s + ']'
                    else:
                        to_eval += '.'
                        to_eval += s
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
        # 'opt_decay_step': 0.1,
        # 'opt_decay_rate': 0.1
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

        node_idx = (feats.sum(dim=-1) != 0.).squeeze()
        adj = adj[node_idx, :]
        adj = adj[:, node_idx]

        feats = feats[node_idx, :]

        centroids = data[0].ndata['centroid'].squeeze()
        centroids = centroids[node_idx, :]

        explanation = adj_to_networkx(adj, feats, rm_iso_nodes=False, centroids=centroids)

        # print('Output:')
        # print('Original Graph: # nodes: {} # edges: {}'.format(
        #     cell_graph.number_of_nodes(),
        #     cell_graph.number_of_edges()
        # ))

        # print('Explanation Graph: # nodes: {} # edges: {}'.format(
        #     explanation.number_of_nodes(),
        #     explanation.number_of_edges()
        # ))

        # 1. visualize the original graph 
        show_cg_flag = load_cell_graph(config['model_type'])
        show_sp_flag = load_superpx_graph(config['model_type'])
        show_sp_map = False
        graph_visualizer = GraphVisualization()
        graph_visualizer(show_cg_flag, show_sp_flag, show_sp_map, data, 1)

        # 2. visualize the explanation graph 
        graph_visualizer = GraphVisualization()
        data = (explanation, data[1], [data[2][0] + '_explanation'])
        graph_visualizer(show_cg_flag, show_sp_flag, show_sp_map, data, 1)


if __name__ == "__main__":
    main(args=parse_arguments())
