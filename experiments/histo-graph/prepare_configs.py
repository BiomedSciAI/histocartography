#!/usr/bin/env python3
import argparse

from histocartography.benchmark.config_generator import ConfigGenerator


def main(args):

    config_generator = ConfigGenerator(
        save_path=args.save_path,
        num_classes=args.num_classes,
        gnn_layer_type=args.gnn_layer_type
    )
    config_generator(model_type=args.model_type)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model_type',
        type=str,
        help='Model type.',
        required=True
    )
    parser.add_argument(
        '-o',
        '--save_path',
        type=str,
        help='Save path.',
        default='../../histocartography/config',
        required=False
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        help='Number of classes (2, 3, 5).',
        default=5,
        required=False
    )
    parser.add_argument(
        '--gnn_layer_type',
        type=str,
        help='GNN layer type (dense_gin_layer or gin_layer).',
        default='gin_layer',
        required=False
    )

    main(parser.parse_args())
