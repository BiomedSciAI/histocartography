import json
import os
import torch
import numpy as np
import pandas as pd 
from PIL import Image
import io
import pickle
import csv
import importlib
from mlflow.pytorch import load_model

from histocartography.ml.models.constants import MODEL_MODULE, AVAILABLE_MODEL_TYPES
from histocartography.interpretability.constants import MODEL_TO_MLFLOW_ID


def plain_model_loading(config):
    model = load_model(MODEL_TO_MLFLOW_ID[config['model_type']][config['explanation_params']['explanation_type']],  map_location=torch.device('cpu'))
    return model


def tentative_model_loading(config):
    # build model from config 
    module = importlib.import_module(MODEL_MODULE.format(config['model_type']))
    model = getattr(module, AVAILABLE_MODEL_TYPES[config['model_type']])(config['model_params'], config['data_params']['input_feature_dims'])

    # buid mlflow model and copy manually the weigths 
    mlflow_model = load_model(MODEL_TO_MLFLOW_ID[config['model_type']][config['explanation_params']['explanation_type']],  map_location=torch.device('cpu'))

    def is_int(s):
        try:
            int(s)
            return True
        except:
            return False

    for n, p in mlflow_model.named_parameters():
        split = n.split('.')
        to_eval = 'model'
        for s in split:
            if is_int(s):
                to_eval += '[' + s + ']'
            else:
                to_eval += '.'
                to_eval += s
        exec(to_eval + '=' + 'p')

    return model


def complete_path(folder, fname):
    """
    Join a folder and a filename
    """
    return os.path.join(folder, fname)


def get_filename(path):
    """
    Get file name in the path
    """
    return os.path.basename(path)


def check_for_dir(path):
    """
    Checks if directory exists, if not, makes a new directory
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_device(cuda=False):
    """
    Get device (cpu or gpu)
    """
    return'cuda:0' if cuda else 'cpu'


def get_files_in_folder(path, extension, with_ext=True):
    """Returns all the file names in a folder, (Relative to the parent folder)
    with a given extension. E.g. if extension == 'svg' it will only return
    svg files.

    Args:
        path (str): path of folder.
        extension (str): type of extension to look for.

    Returns:
        list of file names.
    """
    fnames = [
        f for f in os.listdir(path)
        if os.path.isfile(complete_path(path, f)) and f.endswith(extension)
    ]

    if not with_ext:
        fnames = [f.replace(extension, "") for f in fnames]
    return fnames


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


def save_image(fname, image):
    image.save(fname, quality=95)


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


def load_h5_fnames(base_path, tumor_type, extension, split, fold_id=None):
    """

    :param path:  ../../data/data_split
    :param extension: .h5
    :param split: train
    :return:
    """
    text_path = complete_path(base_path, 'data_split_cv')
    if fold_id is not None:
        text_path = complete_path(text_path, 'data_split_' + str(fold_id + 1))
    fname = split + '_list_' + tumor_type + '.txt'
    h5_files = read_txt(text_path, fname, extension)  # Loads all the .h5 files in the text file
    h5_files.sort(key=lambda x: int(x.split('_')[0]))
    if h5_files is None:
        h5_files = []
    return h5_files


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_checkpoint(model, load_path=''):
    model.load_state_dict(torch.load(load_path))
    return model


def save_checkpoint(model, save_path=''):
    """
    Save a checkpoint model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path))
    

def flatten_dict(d):   
    df = pd.io.json.json_normalize(d, sep='_')
    d = df.to_dict(orient='records')[0]
    return d

    
DATATYPE_TO_SAVEFN = {
    dict: write_json,
    np.ndarray: np.savetxt,
    Image.Image: save_image
}


DATATYPE_TO_EXT = {
    dict: '.json',
    np.ndarray: '.txt',
    Image.Image: '.png'
}
