"""This module handles helper general functions"""
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def fast_mode(input_array: np.array, nr_values, axis: int = 0) -> np.array:
    """Calculates the mode of an tensor over an axis where only values from 0 up to (excluding) nr_values occur.

    Args:
        x (np.array): Input Tensor
        nr_valuesint (int): Possible values. From 0 up to (exclusing) nr_values.
        axis (int, optional): Axis to do the mode over. Defaults to 0.

    Returns:
        np.array: Output Tensor
    """
    output_array = np.empty((nr_values, input_array.shape[1], input_array.shape[2]))
    for i in range(nr_values):
        output_array[i, ...] = (input_array == i).sum(axis=axis)
    return np.argmax(output_array, axis=0)

def fast_histogram(input_array: np.array, nr_values: int) -> np.array:
    """Calculates a histogram of a matrix of the values from 0 up to (excluding) nr_values

    Args:
        x (np.array): Input tensor
        nr_values (int): Possible values. From 0 up to (exclusing) nr_values.

    Returns:
        np.array: Output tensor
    """
    output_array = np.empty(nr_values, dtype=int)
    for i in range(nr_values):
        output_array[i] = (input_array == i).sum()
    return output_array

def read_image(image_path: str) -> np.array:
    """Reads an image from a path and converts it into a numpy array

    Args:
        image_path (str): Path to the image

    Returns:
        np.array: A numpy array representation of the image
    """
    assert image_path.exists()
    img = Image.open(image_path)
    image = np.array(img)
    img.close()
    return image

def start_logging(level="INFO"):
    logging.basicConfig(
        level=level,
        format="%(levelname)s - %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Start logging")

def show_superpixel_heatmap(superpixels):
    fig, ax = plt.subplots(figsize=(14, 14))
    sns.heatmap(superpixels - 1, annot=True, fmt="d", ax=ax, square=True, cbar=False)
    ax.set_axis_off()
    fig.show()

def show_graph(graph):
    nx_G = graph.to_networkx().to_undirected()
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    pos = nx.layout.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
