import math
import torch


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


def compute_normalization_factor(features):
    """
    Compute normalization factors: mean, std of each feature.
    :param features: (list of FloatTensor)
    """
    features = torch.cat([feat for feat in features])
    return {
        'mean': torch.mean(features, dim=0),
        'std': torch.std(features, dim=0)
    }
