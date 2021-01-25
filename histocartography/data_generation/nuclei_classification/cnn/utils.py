import os
import numpy as np
from PIL import Image
import h5py
from matplotlib import pyplot as plt
import seaborn

def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def plot(img, cmap=''):
    if cmap != '':
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.show()


def read_image(path):
    img_ = Image.open(path)
    img = np.array(img_)
    img_.close()
    return img


def read_instance_map(map_path):
    with h5py.File(map_path, 'r') as f:
        instance_map = np.array(f['detected_instance_map']).astype(int)
    return instance_map


def read_centroids(centroids_path, is_label=False):
    with h5py.File(centroids_path, 'r') as f:
        centroids = np.array(f['instance_centroid_location']).astype(int)
        img_dim = np.array(f['image_dimension']).astype(int)

        if is_label:
            labels = np.array(f['instance_centroid_label']).astype(int)
            return centroids, labels, img_dim

    return centroids, img_dim


def save_info(h5_filename, keys, values, dtypes):
    h5_fout = h5py.File(h5_filename, 'w')
    for i in range(len(keys)):
        h5_fout.create_dataset(keys[i], data=values[i], dtype=dtypes[i])
    h5_fout.close()


def plot_confusion_matrix(data, labels, save_path):
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(25, 15))
    plt.title("Confusion Matrix")
    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()























