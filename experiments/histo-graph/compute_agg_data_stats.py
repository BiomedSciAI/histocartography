#!/usr/bin/env python3
"""
Script for computing dataset statistics, e.g., avg number of node per class, image size
"""
import argparse
import numpy as np

from histocartography.utils.io import read_params
from histocartography.dataloader.pascale_dataloader import make_data_loader
from histocartography.dataloader.constants import get_label_to_tumor_type


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

    LABEL_TO_TUMOR_TYPE = get_label_to_tumor_type(config['model_params']['class_split'])

    # make data loaders (train & validation)
    dataloaders, _ = make_data_loader(
        batch_size=1,
        num_workers=0,
        path=args.data_path,
        config=config,
        class_split=config['model_params']['class_split'],
        cuda=False,
        load_cell_graph=True,
        load_superpx_graph=True,
        load_image=True,
        fold_id=0
    )

    image_sizes = []
    cell_nodes = []
    cell_edges = []
    tissue_nodes = []
    tissue_edges = []
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
            cell_edges.append(cell_graph.number_of_edges())
            tissue_nodes.append(tissue_graph.number_of_nodes())
            tissue_edges.append(tissue_graph.number_of_edges())

    image_sizes = np.array(image_sizes)
    cell_nodes = np.array(cell_nodes)
    cell_edges = np.array(cell_edges)
    tissue_nodes = np.array(tissue_nodes)
    tissue_edges = np.array(tissue_edges)
    labels = np.array(labels)

    # 3. compute stats
    for cls in list(set(labels)):
        cn = cell_nodes[labels == cls]
        ce = cell_edges[labels == cls]
        tn = tissue_nodes[labels == cls]
        te = tissue_edges[labels == cls]
        n_px = image_sizes[labels == cls]

        print('Number of samples of class {} is {}'.format(LABEL_TO_TUMOR_TYPE[cls], len(cn)))

        print('Average number of cell nodes of class {} is {} +/- {}. Max/Min ratio is {}'.format(
            LABEL_TO_TUMOR_TYPE[cls],
            np.mean(cn),
            np.std(cn),
            np.min(cn) / np.max(cn)
        ))

        print('Average number of cell edges of class {} is {} +/- {}. Max/Min ratio is {}'.format(
            LABEL_TO_TUMOR_TYPE[cls],
            np.mean(ce),
            np.std(ce),
            np.min(ce) / np.max(ce)
        ))

        print('Average number of tissue nodes of class {} is {} +/- {}. Max/Min ratio is {}'.format(
            LABEL_TO_TUMOR_TYPE[cls],
            np.mean(tn),
            np.std(tn),
            np.min(tn) / np.max(tn)
        ))

        print('Average number of tissue edges of class {} is {} +/- {}. Max/Min ratio is {}'.format(
            LABEL_TO_TUMOR_TYPE[cls],
            np.mean(te),
            np.std(te),
            np.min(te) / np.max(te)
        ))

        print('Average number of pixels of class {} is {} +/- {}. Max/Min ratio is {}'.format(
            LABEL_TO_TUMOR_TYPE[cls],
            np.mean(n_px),
            np.std(n_px),
            np.min(n_px) / np.max(n_px)
        ))

    # 4. overall stats
    print('Number of samples {}'.format(len(cell_nodes)))

    print('Average number of cell nodes is {} +/- {}. Max/Min ratio is {}'.format(
        np.mean(cell_nodes),
        np.std(cell_nodes),
        np.min(cell_nodes) / np.max(cell_nodes)
    ))

    print('Average number of cell edges is {} +/- {}. Max/Min ratio is {}'.format(
        np.mean(cell_edges),
        np.std(cell_edges),
        np.min(cell_edges) / np.max(cell_edges)
    ))

    print('Average number of tissue nodes is {} +/- {}. Max/Min ratio is {}'.format(
        np.mean(tissue_nodes),
        np.std(tissue_nodes),
        np.min(tissue_nodes) / np.max(tissue_nodes)
    ))

    print('Average number of tissue edges is {} +/- {}. Max/Min ratio is {}'.format(
        np.mean(tissue_edges),
        np.std(tissue_edges),
        np.min(tissue_edges) / np.max(tissue_edges)
    ))

    print('Average number of pixels of class is {} +/- {}. Max/Min ratio is {}'.format(
        np.mean(image_sizes),
        np.std(image_sizes),
        np.max(image_sizes) / np.min(image_sizes)
    ))


if __name__ == "__main__":
    main(args=parse_arguments())
