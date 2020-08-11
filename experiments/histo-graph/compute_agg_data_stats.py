#!/usr/bin/env python3
"""
Script for computing dataset statistics, e.g., avg number of node per class, image size
"""
import argparse
import numpy as np

from histocartography.utils.io import read_params
from histocartography.dataloader.pascale_dataloader import make_data_loader
from histocartography.dataloader.constants import LABEL_TO_TUMOR_TYPE


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_path',
        type=str,
        help='path to the data.',
        default='../../data/',
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

    parser.add_argument(
        '-o',
        '--out_folder',
        type=str,
        help='where to save the stats.',
        default='',
        required=False
    )

    return parser.parse_args()


def main(args):
    """
    Train HistoGraph.
    Args:
        args (Namespace): parsed arguments.
    """

    # load config file
    config = read_params(args.config_fpath, verbose=True)

    # make data loaders (train & validation)
    dataloaders, _ = make_data_loader(
        batch_size=1,
        num_workers=0,
        path=args.data_path,
        config=config,
        load_cell_graph=True,
        load_superpx_graph=True,
        load_image=True
    )

    image_sizes = []
    cell_nodes = []
    tissue_nodes = []
    labels = []

    counter = 0
    for split in ['train', 'val', 'test']:
        for data, label in dataloaders[split]:

            print('Processing samples:', counter)
            counter += 1

            # 1. extract data
            cell_graph = data[0]
            tissue_graph = data[1]
            image = data[3][0]

            # 2. append data
            labels.append(label.item())
            width, height = image.size
            image_sizes.append(width * height)
            cell_nodes.append(cell_graph.number_of_nodes())
            tissue_nodes.append(tissue_graph.number_of_nodes())

    image_sizes = np.array(image_sizes)
    cell_nodes = np.array(cell_nodes)
    tissue_nodes = np.array(tissue_nodes)
    labels = np.array(labels)

    # 3. compute stats
    for cls in list(set(labels)):
        cn = cell_nodes[labels == cls]
        tn = tissue_nodes[labels == cls]
        n_px = image_sizes[labels == cls]

        print('Number of samples of class {}'.format(len(cn)))

        print('Average number of cell nodes of class {} is {}'.format(
            LABEL_TO_TUMOR_TYPE[str(cls)],
            np.mean(cn)
        ))

        print('Average number of tissue nodes of class {} is {}'.format(
            LABEL_TO_TUMOR_TYPE[str(cls)],
            np.mean(tn)
        ))

        print('Average number of pixels of class {} is {}'.format(
            LABEL_TO_TUMOR_TYPE[str(cls)],
            np.mean(n_px)
        ))

    # 4. overall stats
    print('Number of samples of class {}'.format(len(cell_nodes)))

    print('Average number of cell nodes is {}'.format(
        np.mean(cell_nodes)
    ))

    print('Average number of tissue nodes is {}'.format(
        np.mean(tissue_nodes)
    ))

    print('Average number of pixels of class is {}'.format(
        np.mean(image_sizes)
    ))


if __name__ == "__main__":
    main(args=parse_arguments())
