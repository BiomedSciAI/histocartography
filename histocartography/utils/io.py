import json
import os
import torch
import numpy as np
import PIL
from PIL import Image
import io
import pickle
import csv
import requests


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


def h5_to_tensor(h5_object, device):
    """
    Convert h5 object into torch tensor
    """
    tensor = torch.from_numpy(np.array(h5_object[()])).to(device)
    return tensor


def h5_to_numpy(h5_object):
    """
    Convert h5 object into numpy array
    """
    out = np.array(h5_object[()])
    return out


def load_json(fname):
    """
    Load json file as a dict.
    :param fname: (str) path to json
    """
    with open(fname, 'r') as in_config:
        params = json.load(in_config)
    return params


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


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


def download_test_data(out_dir):
    # 1. download 283 dcis 4 cell graph:
    fname = os.path.join(out_dir, 'cell_graphs', '283_dcis_4.bin')
    download_box_link(
        'https://ibm.box.com/shared/static/nuxuhc1upe0x1mq2t7njtl7hp5m7x67f.bin',
        fname)

    # 2. download 283 dcis 4 tissue graph:
    fname = os.path.join(out_dir, 'tissue_graphs', '283_dcis_4.bin')
    download_box_link(
        'https://ibm.box.com/shared/static/cw2z7mu6n7b9mlybb83k65sn7dre0y9k.bin',
        fname)

    # 3. download test images:
    fname = os.path.join(out_dir, 'images', '17B0031061.png')
    download_box_link(
        'https://ibm.box.com/shared/static/yzyrb051125k866ehaowe4gyep90wws6.png',
        fname)

    fname = os.path.join(out_dir, 'images', '18B000646H.png')
    download_box_link(
        'https://ibm.box.com/shared/static/pyj5igmqzqyezpkk4d347a7e1uw3l6y3.png',
        fname)

    fname = os.path.join(out_dir, 'images', '283_dcis_4.png')
    download_box_link(
        'https://ibm.box.com/shared/static/r9ad48jn974e9xtpztk72qfplcg5nv5g.png',
        fname)

    fname = os.path.join(out_dir, 'images', '283_dcis_4_annotation.png')
    download_box_link(
        'https://ibm.box.com/shared/static/ynprh6b9naggb2bmufdd762z8ui2wmbf.png',
        fname)

    fname = os.path.join(out_dir, 'images', '283_dcis_4_background.png')
    download_box_link(
        'https://ibm.box.com/shared/static/03kh4umccufuw2t0mott39d2c01jjj14.png',
        fname)

    fname = os.path.join(out_dir, 'images', '16B0001851_Block_Region_3.jpg')
    download_box_link(
        'https://ibm.box.com/shared/static/jkut7hsigpg278xsoh764bguwehuwd5f.jpg',
        fname)

    # 4. download nuclei maps:
    fname = os.path.join(out_dir, 'nuclei_maps', '283_dcis_4.h5')
    download_box_link(
        'https://ibm.box.com/shared/static/qdvvbz0ninnqhic2k5b7wpsshh666mm4.h5',
        fname)


def download_example_data(out_dir=''):

    fname = os.path.join(out_dir, 'images', '283_dcis_4.png')
    download_box_link(
        'https://ibm.box.com/shared/static/r9ad48jn974e9xtpztk72qfplcg5nv5g.png',
        fname)

    fname = os.path.join(out_dir, 'images', '1238_adh_10.png')
    download_box_link(
        'https://ibm.box.com/shared/static/q2h5b19ay3hklf74acft1fn60i2746y2.png',
        fname)

    fname = os.path.join(out_dir, 'images', '1937_benign_4.png')
    download_box_link(
        'https://ibm.box.com/shared/static/lskhxrttnbpg3eoxon2j6mxrjceqr4cd.png',
        fname)
