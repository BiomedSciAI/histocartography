import math
import torch
import h5py

from histocartography.utils.io import complete_path, h5_to_tensor


def compute_box_centroid(box):
    """
    Compute the centroid of a bounding box
    """
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def compute_l2_distance(pt1, pt2):
    """
    Compute l2 distance between pt1 and pt2
    Args:
        :param pt1: (list)
        :param pt2: (list)
    """
    squares = [(p - q) ** 2 for p, q in zip(pt1, pt2)]
    return sum(squares) ** .5


def compute_edge_weight(dist):
    """
    Compute the edge weight given a distance
    Args:
        :param dist: normalised distance
    """
    # @TODO add decaying parameter ?
    return math.exp(-dist)


def compute_normalization_factor(path, fnames):
    """
    Compute normalization factors: mean, std of each feature.
    :param features: (list of FloatTensor)
    """
    features = []
    for fname in fnames:
        with h5py.File(complete_path(path, fname), 'r') as f:
            features.append(h5_to_tensor(f['instance_features'], device='cpu'))
            f.close()

    features = torch.cat([feat for feat in features])
    return {
        'mean': torch.mean(features, dim=0),
        'std': torch.std(features, dim=0)
    }


def compute_norm(fnames):
    sum_ = torch.zeros([32], dtype=torch.float)  # torch.FloatTensor([0.0])
    std_ = torch.zeros([32], dtype=torch.float)
    num_nodes = 0

    for fname in fnames:
        with h5py.File(fname, 'r') as f:
            node_features = h5_to_tensor(f['vae_embeddings'], device='cpu')
            sum_ += torch.sum(node_features, dim=0)
            std_ += torch.std(node_features, dim=0)
            num_nodes += node_features.size(0)
            f.close()
    mean = sum_ / num_nodes
    std_ /= len(fnames)

    return {
            'mean': mean,
            'std_dev': std_
    }
