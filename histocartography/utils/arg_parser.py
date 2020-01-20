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
        default=1,
        required=False
    )
    parser.add_argument(
        '-r',
        '--train_ratio',
        type=int,
        help='% of data to use for training.',
        default=0.8,
        required=False
    )
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        help='model name.',
        default='model',
        required=False
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        help='batch size.',
        default=8,
        required=False
    )
    parser.add_argument(
        '--epochs', type=int, help='epochs.', default=10, required=False
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
        '--text_path',
        type=str,
        help='Path to folder where train:test:val split is located',
        default='',

    )

    return parser.parse_args()
