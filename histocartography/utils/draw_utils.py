import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')


def draw_ellipse(centroid, draw, fill_col, size=5, outline=(0, 0, 255)):
    thickness = 2
    for i in range(thickness):
        draw.ellipse((centroid[0] - size -i, centroid[1] - size -i, centroid[0] + size + i, centroid[1] + size + i),
                 fill=fill_col,
                 outline=outline)


def draw_large_circle(centroid, draw):
    draw.ellipse((centroid[0] - 25, centroid[1] - 25, centroid[0] + 25, centroid[1] + 25),
                 outline='blue')


def draw_line(source_centroid, dest_centroid, draw, fill_col, line_wid):
    draw.line((source_centroid[0], source_centroid[1], dest_centroid[0], dest_centroid[1]),
              fill=fill_col,
              width=line_wid)


def draw_poly(xy, draw, outline=None, fill=None):
    draw.polygon(xy, outline=outline, fill=fill)


def rgb(minimum, maximum, value, transparency=None):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    if transparency is not None:
        return (r, g, b, transparency)
    return (r, g, b)


def plot_tSNE(x, labels, save_fname, label_to_name):
    """
    Plot tSNE
    :param x:
    :param labels:
    :return:
    """
    colors = ['red', 'green', 'blue', 'purple', 'cyan']
    plt.scatter(x[:, 0], x[:, 1], marker='x', c=labels, cmap=matplotlib.colors.ListedColormap(colors))

    cb = plt.colorbar()
    loc = np.arange(0, max(labels), max(labels) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(np.unique(labels))

    plt.savefig(save_fname)
