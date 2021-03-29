import json
import os
import torch
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import io
import pickle
import csv
import importlib
import requests


def get_device(cuda=False):
    """
    Get device (cpu or gpu)
    """
    return 'cuda:0' if cuda else 'cpu'


def is_mlflow_url(candidate):
    # is it stored on s3 ?
    if candidate.find('s3://mlflow/') != -1:
        return True
    # is it a remote mlflow run ?
    if candidate.find('runs:/') != -1:
        return True
    # is it a remote mlflow model ?
    if candidate.find('models:/') != -1:
        return True
    # is it a local run
    if os.path.exists(os.path.join(candidate, 'MLmodel')):
        return True
    return False


def is_box_url(candidate):
    # check if IBM box static link
    if 'https://ibm.box.com/shared/static/' in candidate:
        return True
    return False


def buffer_plot_and_get(fig):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=200)
    buf.seek(0)
    return PIL.Image.open(buf)


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
    if path and not os.path.exists(path):
        os.makedirs(path)


def get_device(cuda=False):
    """
    Get device (cpu or gpu)
    """
    return 'cuda:0' if cuda else 'cpu'


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
        print('Fname:', fname, dir, extension)
        files = f.read()
        print('Files are:', files)
        files = files.split()
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
    h5_files = read_txt(
        text_path, fname, extension
    )  # Loads all the .h5 files in the text file
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


def download_box_link(url, out_fname='box.file'):
    out_dir = os.path.dirname(out_fname)
    check_for_dir(out_dir)
    if os.path.isfile(out_fname):
        print('File already downloaded.')
        return out_fname

    r = requests.get(url, stream=True)

    with open(out_fname, "wb") as large_file:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                large_file.write(chunk)
    return out_fname


DATATYPE_TO_SAVEFN = {
    dict: write_json,
    np.ndarray: np.savetxt,
    Image.Image: save_image
}

DATATYPE_TO_EXT = {dict: '.json', np.ndarray: '.txt', Image.Image: '.png'}


def download_test_data(out_dir):
    # 1. download 283 dcis 4 cell graph:
    fname = os.path.join(out_dir, 'cell_graphs', '283_dcis_4.bin')
    download_box_link('https://ibm.box.com/shared/static/nuxuhc1upe0x1mq2t7njtl7hp5m7x67f.bin', fname)

    # 2. download 283 dcis 4 tissue graph:
    fname = os.path.join(out_dir, 'tissue_graphs', '283_dcis_4.bin')
    download_box_link('https://ibm.box.com/shared/static/cw2z7mu6n7b9mlybb83k65sn7dre0y9k.bin', fname)

    # 3. download test images:
    fname = os.path.join(out_dir, 'images', '17B0031061.png')
    download_box_link('https://ibm.box.com/shared/static/yzyrb051125k866ehaowe4gyep90wws6.png', fname)

    fname = os.path.join(out_dir, 'images', '18B000646H.png')
    download_box_link('https://ibm.box.com/shared/static/pyj5igmqzqyezpkk4d347a7e1uw3l6y3.png', fname)

    fname = os.path.join(out_dir, 'images', '283_dcis_4.png')
    download_box_link('https://ibm.box.com/shared/static/r9ad48jn974e9xtpztk72qfplcg5nv5g.png', fname)

    fname = os.path.join(out_dir, 'images', '16B0001851_Block_Region_3.jpg')
    download_box_link('https://ibm.box.com/shared/static/jkut7hsigpg278xsoh764bguwehuwd5f.jpg', fname)
