import math


def compute_box_centroid(box):
    """
    Compute the centroid of a bounding box
    Args:
        :param box: @TODO agree on the bounding box format.
        :return: centroid
    """
    return [(box[0]+box[2])/2, (box[1]+box[3])/2]


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
    # @TODO check what is the best dist to edge weight normalization.
    return math.exp(dist)
