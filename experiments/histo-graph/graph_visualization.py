#!/usr/bin/env python3
"""
Script for visualizing graph-based histocartography models
"""
import logging
import sys
import argparse
import dgl

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
        default='/Users/frd/Documents/Code/Projects/Experiments/data_dummy_sp/',
        required=False
    )

    parser.add_argument(
        '-conf',
        '--config_fpath',
        type=str,
        help='path to the config file.',
        default='../../histocartography/config/concat_graph_model_config/concat_graph_model_config_0.json',
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
        batch_size=2,
        num_workers=0,
        path=args.graph_data_path,
        config=config,
        load_cell_graph=load_cell_graph(config['model_type']),
        load_superpx_graph=load_superpx_graph(config['model_type']),
        load_image=True
    )

    graph_visualizer = GraphVisualization()
    model_type = config['model_type']

    if model_type == 'multi_level_graph_model' or 'concat_graph_model':
        for (cell_graph, superpx_graph, superpx_map, assign_mat, image, image_name), label in dataloaders['train']:
            graph_visualizer(image[0], image_name[0], model_type, dgl.unbatch(cell_graph)[0],
                             dgl.unbatch(superpx_graph)[0], superpx_map[0])
    elif model_type == 'cell_graph_model':
        for (graph, image, image_name), label in dataloaders['train']:
            graph_visualizer(image[0], image_name[0], model_type, dgl.unbatch(graph)[0])
    elif model_type == 'superpx_graph_model':
        for (graph, sp_mask, image, image_name), label in dataloaders['train']:
            graph_visualizer(image[0], image_name[0], model_type, dgl.unbatch(graph)[0], sp_mask[0])


if __name__ == "__main__":
    main(args=parse_arguments())
