import numpy as np
import os
from histocartography.utils.io import check_for_dir
from dgl.data.utils import save_graphs
import torch


def feature_normalization(centroids, features, image_dim, transformer=None, device='cpu'):
    #image_dim = image_dim[[1, 0]]  # flipping image size to be [y,x]

    image_dim = image_dim.type(torch.float32)
    image_dim = torch.index_select(image_dim, 0, torch.LongTensor([1, 0]))
    image_dim = image_dim.type(torch.float32)
    norm_centroid = centroids / image_dim

    # if transformer is not None:
    #     # normalize the cell features @TODO: by pass normalization 
    #     features = \
    #         ((features - transformer['mean'].to(device)) / (transformer['std'].to(device)))

    features = torch.cat((features, norm_centroid), dim=1)
    return features


def build_graph(builder_type, config,  *args):
    graph_built = builder_type(args[0], args[1], args[2])
    return graph_built


def save_graph_file(path_to_save, tumor_type, config, feature_used,  filename, graph):

    dgl_file = os.path.splitext(filename)[0] + '.bin'
    builder_graph = config['model_type'].replace('_model', '_builder')
    path = os.path.join(path_to_save, 'graphs', config['model_type'], feature_used, tumor_type)
    check_for_dir(path)
    save_graphs(os.path.join(path, dgl_file), graph)


def get_data_aug_sp(indices, start_coord, old_centroids, old_features, new_indices, new_centroids, new_features,
                     superpx_map, crop_dim):

    n = 0
    centroids = np.zeros((len(indices), 2))
    features = np.zeros((len(indices), old_features.shape[1])) ##TODO : need to check

    for ind in range(len(indices)):
        if indices[ind] in new_indices:
            centroids[ind] = new_centroids[n]
            features[ind] = new_features[n]
            n += 1
        else:
            centroids[ind] = np.array([old_centroids[ind] - start_coord])
            features[ind] = np.array([old_features[ind]])
    sp_map = superpx_map[start_coord[1]:start_coord[1] + crop_dim[0], start_coord[0]:start_coord[0] + crop_dim[1]]

    return features, centroids, sp_map


def get_data_aug(indices, start_coord, centroids, features):
    centroids = np.array([centroids[x - 1] - start_coord for x in indices])
    features = np.array([features[x - 1] for x in indices])
    return features, centroids

def save_mat(path_to_save, tumor_type, config, filename, mat):
    npy_file = os.path.splitext(filename)[0] + '.npy'
    builder_graph = config['model_type'].replace('_model', '_builder')
    path = os.path.join(path_to_save, 'assignment_mat', tumor_type)
    check_for_dir(path)
    np.save(os.path.join(path, npy_file), mat)


