import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_path',
        type=str,
        help='path to the data.',
        default='data/',
        required=True
    )
    parser.add_argument(
        '-conf',
        '--config_fpath',
        type=str,
        help='path to the config file.',
        default='',
        required=True
    )
    parser.add_argument(
        '-p',
        '--number_of_workers',
        type=int,
        help='number of workers.',
        default=0,
        required=False
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='path to where the model is saved.',
        default='trained_model',
        required=False
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        help='batch size.',
        default=2,
        required=False
    )
    parser.add_argument(
        '--epochs', type=int, help='epochs.', default=2, required=False
    )
    parser.add_argument(
        '-l',
        '--learning_rate',
        type=float,
        help='learning rate.',
        default=10e-3,
        required=False
    )
    parser.add_argument(
        '--visualization',
        type=bool,
        help='True if visualisation of graphs',
        default=True

    )

    return parser.parse_args()
