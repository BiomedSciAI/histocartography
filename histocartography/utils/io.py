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


def split_path(path):
    text_path = path
    tumor_name = ""
    for i in range(3):
        text_path = os.path.dirname(text_path)
        if i == 0:
            tumor_name = os.path.basename(text_path)
    return text_path, tumor_name

def get_files_from_text(path, extension, split):

    text_path, tumor_type = split_path(path)

    # get tumor name
    tumor = [token for token in tumor_type.split('_') if not token.isdigit()]
    tumor = '_' + ''.join(map(str, tumor))

    # get text path
    text_path = complete_path(text_path, 'data_split')
    list_of_files = get_files_in_folder(text_path, 'txt')  # lists all files in text_path(all text files)

    # returns relevant text file to be read
    read_file = list(filter(lambda x: tumor in x and split in x, list_of_files))
    h5_files = read_txt(text_path, read_file[0], extension)  # Loads all the .h5 files in the text file

    return [complete_path(path, g) for g in h5_files]
