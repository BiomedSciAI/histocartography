#!/usr/bin/env python3
"""
Script for generating explanations
"""

import importlib
import torch
import mlflow
import numpy as np
from tqdm import tqdm 
import os 

from histocartography.utils.io import read_params
from histocartography.dataloader.pascale_dataloader import make_data_loader
from histocartography.ml.models.constants import AVAILABLE_MODEL_TYPES, MODEL_TYPE
from histocartography.interpretability.constants import AVAILABLE_EXPLAINABILITY_METHODS, INTERPRETABILITY_MODEL_TYPE_TO_LOAD_FN
from histocartography.utils.arg_parser import parse_arguments
from histocartography.ml.models.constants import load_superpx_graph, load_cell_graph
from histocartography.utils.io import get_device, flatten_dict
from histocartography.dataloader.constants import get_label_to_tumor_type
from histocartography.utils.visualization import tSNE
from histocartography.utils.draw_utils import plot_tSNE


# flush warnings
import warnings
warnings.filterwarnings("ignore")

# cuda support
CUDA = torch.cuda.is_available()
DEVICE = get_device(CUDA)

# base save path
BASE_SAVE_PATH = '/dataT/pus/histocartography/Data/explainability/output/gnn/latent'


def main(args):
    """
    Train HistoGraph.
    Args:
        args (Namespace): parsed arguments.
    """

    # load config file
    config = read_params(args.config_fpath, verbose=True)

    # constants
    label_to_tumor_type = get_label_to_tumor_type(config['model_params']['class_split'])

    # extract interpretability model type
    interpretability_model_type = config['explanation_params']['explanation_type']

    # make data loaders
    dataloaders, input_feature_dims = make_data_loader(
        batch_size=1,
        num_workers=args.number_of_workers,
        path=args.data_path,
        config=config,
        class_split=config['model_params']['class_split'],
        cuda=CUDA,
        load_cell_graph=load_cell_graph(config['model_type']),
        load_superpx_graph=load_superpx_graph(config['model_type']),
        load_image=True,
        load_nuclei_seg_map=True,
        fold_id=0
    )

    # append dataset info to config
    config['data_params'] = {}
    config['data_params']['input_feature_dims'] = input_feature_dims

    # define GNN model
    if interpretability_model_type in list(AVAILABLE_EXPLAINABILITY_METHODS.keys()):
        module = importlib.import_module('histocartography.utils.io')
        model = getattr(module, INTERPRETABILITY_MODEL_TYPE_TO_LOAD_FN[interpretability_model_type])(config)
        if CUDA:
            model = model.cuda()
    else:
        raise ValueError(
            'Model: {} not recognized. Options are: {}'.format(
                model_type, list(AVAILABLE_EXPLAINABILITY_METHODS.keys())
            )
        )

    # set flag to enable forward hook
    model.set_forward_hook(model.pred_layer.mlp, '0')  # hook before the last classification layer

    # define dimensionality reduction algorithm
    tsne = tSNE()

    # mlflow log parameters
    inter_config = flatten_dict(config['explanation_params'])
    for key, val in inter_config.items():
        mlflow.log_param(key, val)

    # run forward pass to extract the embeddings 
    all_latent_representations = []
    all_labels = []
    counter = 0
    for data, label in tqdm(dataloaders[args.split]):
        logits = model(data)
        all_latent_representations.append(model.latent_representation)
        all_labels.append(label.item())
        counter += 1
        # if counter > 10:
        #     break
    all_latent_representations = torch.stack(all_latent_representations, dim=0).cpu().detach().numpy().squeeze()

    # run t-SNE and plot
    latent_embedded = tsne(all_latent_representations)

    # plot and save 
    plot_tSNE(latent_embedded, all_labels, os.path.join(BASE_SAVE_PATH, 'tsne.png'), label_to_tumor_type)


if __name__ == "__main__":
    main(args=parse_arguments())
