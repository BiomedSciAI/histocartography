import numpy as np
from matplotlib import pyplot as plt

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

