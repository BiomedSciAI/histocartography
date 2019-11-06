import json


def read_params(fname):
    """
    Config file
    :param fname:
    :param verbose:
    :return: config params
    """
    with open(fname, 'r') as in_config:
        config_params = json.load(in_config)
    return config_params
