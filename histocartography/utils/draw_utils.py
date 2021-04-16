import random
import matplotlib
import matplotlib.pyplot as plt
from collections.abc import Iterable
import numpy as np

plt.style.use("seaborn-whitegrid")


def name2rgb(color_name):
    return tuple(matplotlib.colors.to_rgba(color_name)[0:3])


def map_value_to_color(value, colormap, number_of_colors=256, **kwargs):
    cmap = matplotlib.cm.get_cmap(colormap, number_of_colors)
    if not isinstance(value, str):
        value = 255 * np.array(cmap(value, **kwargs))
        value = tuple(value.astype(int))
    return value


def draw_ellipse(centroid, draw, fill_col, size=5, outline=(0, 0, 255)):
    thickness = 2
    for i in range(thickness):
        draw.ellipse(
            (
                centroid[0] - size - i,
                centroid[1] - size - i,
                centroid[0] + size + i,
                centroid[1] + size + i,
            ),
            fill=fill_col,
            outline=outline,
        )


def draw_large_circle(centroid, draw):
    draw.ellipse(
        (centroid[0] - 25,
         centroid[1] - 25,
         centroid[0] + 25,
         centroid[1] + 25),
        outline="blue",
    )


def draw_circle(
        centroid,
        draw,
        radius=5,
        outline_color="yellow",
        fill_color="yellow",
        width=2):
    draw.ellipse(
        (
            centroid[0] - radius,
            centroid[1] - radius,
            centroid[0] + radius,
            centroid[1] + radius,
        ),
        outline=outline_color,
        fill=fill_color,
        width=width,
    )


def draw_line(source_centroid, dest_centroid, draw, fill_col, line_wid):
    draw.line(
        (source_centroid[1],
         source_centroid[0],
         dest_centroid[1],
         dest_centroid[0]),
        fill=fill_col,
        width=line_wid,
    )


def draw_poly(xy, draw, outline=None, fill=None):
    draw.polygon(xy, outline=outline, fill=fill)


def rgb(minimum, maximum, value, transparency=None):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    if transparency is not None:
        return (r, g, b, transparency)
    return (r, g, b)
