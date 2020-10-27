import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import h5py
import os

def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def plot(img, cmap=''):
    if cmap == '':
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    plt.show()


def normalize(w):
    minm = np.min(w)
    maxm = np.max(w)
    if maxm - minm != 0:
        w = (w - minm) / (maxm - minm)
    return w


def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return '#{:02x}{:02x}{:02x}'.format( r, g, b)


def read_image(path):
    img_ = Image.open(path)
    img = np.array(img_)
    img_.close()
    return img


def read_instance_map(path):
    with h5py.File(path, 'r') as f:
        nuclei_instance_map = np.array(f['detected_instance_map'])
    return nuclei_instance_map


def read_features(path):
    with h5py.File(path, 'r') as f:
        features = np.array(f['embeddings'])
    return features


def read_info(path):
    with h5py.File(path, 'r') as f:
        centroids = np.array(f['instance_centroid_location']).astype(int)
        labels = np.array(f['instance_centroid_label']).astype(int)
    return centroids, labels


