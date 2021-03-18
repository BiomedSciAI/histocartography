import argparse


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path',
        type=str,
        help='path to the data.',
        default='/dataT/frd/Code/graph_building/data/',
        required=False
    )
    parser.add_argument(
        '--save_path',
        type=str,
        help='Path where graphs saved',
        default='/dataT/frd/Code/graph_building/data/graphs/',
        required=False
    )
    parser.add_argument(
        '--features_used',
        type=str,
        help='hand_crafted(features_hc) or features_cnn',
        required=True
    )
    parser.add_argument(
        '--configuration',
        type=str,
        help='json configuration file with parameters',
        required=True
    )

    return parser.parse_args()

