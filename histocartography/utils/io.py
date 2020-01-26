import json
import os
import torch
import numpy as np
from PIL import Image


def complete_path(folder, fname):
    """
    Join a folder and a filename
    """
    return os.path.join(folder, fname)


def get_device(cuda=False):
    """
    Get device (cpu or gpu)
    """
    return'cuda:0' if cuda else 'cpu'


def get_files_in_folder(path, extension):
    """Returns all the file names in a folder, (Relative to the parent folder)
    with a given extension. E.g. if extension == 'svg' it will only return
    svg files.

    Args:
        path (str): path of folder.
        extension (str): type of extension to look for.

    Returns:
        list of file names.
    """
    return [
        f for f in os.listdir(path)
        if os.path.isfile(complete_path(path, f)) and f.endswith(extension)
    ]


def get_dir_in_folder(path):
    return [f.name for f in os.scandir(path) if f.is_dir()]
    # return [f for f in os.listdir(path) if f.is_dir()]


def h5_to_tensor(h5_object, device):
    """
    Convert h5 object into torch tensor
    """
    tensor = torch.from_numpy(np.array(h5_object[()])).to(device)
    return tensor


def load_json(fname):
    """
    Load json file as a dict.
    :param fname: (str) path to json
    """
    with open(fname, 'r') as in_config:
        config_params = json.load(in_config)
    return config_params


def load_image(fname):
    """
    Load an image as a PIL image

    Args:
        :param fname: (str) path to image
    """
    image = Image.open(fname)
    return image


def save_image(image, fname):
    image.save(fname)


def show_image(image):
    """
    Show a PIL image
    """
    image.show()


def read_params(fname, verbose=False):
    """
    Config file contains either a simple config set or a list of configs
        (used to run several experiments).

    Args:
        :param fname:
        :param reading_index:
        :param verbose:
        :return: config params
    """
    with open(fname, 'r') as in_config:
        config_params = json.load(in_config)
        if verbose:
            print('\n*** Model config parameters:', config_params)
    return config_params


def read_txt(dir, fname, extension):
    """
   Reads files from a text file and adds the extension for each file name in the text
    """
    with open(complete_path(dir, fname)) as f:
        files = f.read().split()
        files = [x + extension for x in files]
    return files


def load_h5_fnames(base_path, tumor_type, extension, split):
    """

    :param path:  ../../data/data_split
    :param extension: .h5
    :param split: train
    :return:
    """
    text_path = complete_path(base_path, 'data_split')
    fname = split + '_list_' + tumor_type + '.txt'
    h5_files = read_txt(text_path, fname, extension)  # Loads all the .h5 files in the text file
    return h5_files
