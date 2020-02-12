#!/usr/bin/env python3
"""
Script for visualizing graph-based histocartography models
"""
import logging
import sys
import argparse


from histocartography.utils.io import read_params
from histocartography.dataloader.pascale_dataloader import make_data_loader
from histocartography.utils.visualization import GraphVisualization
from histocartography.ml.models.constants import load_superpx_graph, load_cell_graph


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--graph_data_path',
        type=str,
        help='path to the graph data.',
        default='../../pascale/',
        required=False
    )

    parser.add_argument(
        '-conf',
        '--config_fpath',
        type=str,
        help='path to the config file.',
        default='../../histocartography/config/cell_graph_model_config/cell_graph_model_config_0.json',
        required=False
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        required=False
    )

    return parser.parse_args()


# setup logging
log = logging.getLogger('Histocartography::Training')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
h1.setFormatter(formatter)
log.addHandler(h1)


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
        num_workers=0,
        path=args.graph_data_path,
        config=config,
        load_cell_graph=load_cell_graph(config['model_type']),
        load_superpx_graph=load_superpx_graph(config['model_type']),
        load_image=True,
        show_superpx=load_superpx_graph(config['model_type'])
    )

    graph_visualizer = GraphVisualization()

    show_cg_flag = load_cell_graph(config['model_type'])
    show_sp_flag = load_superpx_graph(config['model_type'])

    for data, label in dataloaders['test']:
        graph_visualizer(show_cg_flag, show_sp_flag, data, args.batch_size)


if __name__ == "__main__":
    main(args=parse_arguments())
