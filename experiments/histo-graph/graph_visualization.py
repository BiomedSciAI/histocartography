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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--graph_data_path',
        type=str,
        help='path to the graph data.',
        default='../../data/graphs',
        required=False
    )
    parser.add_argument(
        '-img',
        '--image_data_path',
        type=str,
        help='path to the images.',
        default='../../data/images',
        required=False
    )
    parser.add_argument(
        '-conf',
        '--config_fpath',
        type=str,
        help='path to the config file.',
        default='../../histocartography/config/cell_graph_config_file.json',
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
        batch_size=1,
        train_ratio=1.0,
        num_workers=0,
        path=args.graph_data_path,
        config=config,
        img_path=args.image_data_path
    )

    graph_visualizer = GraphVisualization()

    for (graph, image, image_name), label in dataloaders['train']:
        graph_visualizer(dgl.unbatch(graph)[0], image[0], image_name[0])


if __name__ == "__main__":
    main(args=parse_arguments())
