import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')


def draw_ellipse(centroid, draw, fill_col):
    draw.ellipse((centroid[0] - 5, centroid[1] - 5, centroid[0] + 5, centroid[1] + 5),
                 fill=fill_col,
                 outline=(0, 0, 0))


def draw_large_circle(centroid, draw):
    draw.ellipse((centroid[0] - 25, centroid[1] - 25, centroid[0] + 25, centroid[1] + 25),
                 outline='blue')


def draw_line(source_centroid, dest_centroid, draw, fill_col, line_wid):
    draw.line((source_centroid[0], source_centroid[1], dest_centroid[0], dest_centroid[1]),
              fill=fill_col,
              width=line_wid)


def draw_poly(xy, draw):
    draw.polygon(xy, outline=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 100))


def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
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
