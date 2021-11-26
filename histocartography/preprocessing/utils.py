"""Preprocessing utilities"""
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import openslide


def fast_histogram(input_array: np.ndarray, nr_values: int) -> np.ndarray:
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


def load_image(image_path: Path) -> np.ndarray:
    """Loads an image from a given path and returns it as a numpy array

    Args:
        image_path (Path): Path of the image

    Returns:
        np.ndarray: Array representation of the image
    """
    assert image_path.exists()
    try:
        with Image.open(image_path) as img:
            image = np.array(img)
    except OSError as e:
        logging.critical("Could not open %s", image_path)
        raise OSError(e)
    return image


def load_wsi(image_path: str, downsample_level: int = 0) -> np.ndarray:
    """Loads a whole-slide image (WSI) from a given path and returns it at
    a desired downsample level as a numpy array.

    Args:
        image_path (str):       Path of the WSI
        downsample_level (int): Downsampling level (not factor)

    Returns:
        np.ndarray: Array representation of the WSI
    """
    assert Path(image_path).exists()
    try:
        wsi = openslide.open_slide(image_path)
        assert downsample_level < wsi.level_count
        level_dims = wsi.level_dimensions
        wsi = np.array(wsi.read_region(location=(0, 0),
                                       level=downsample_level,
                                       size=level_dims[downsample_level]).convert('RGB'))
    except OSError as e:
        logging.critical("Could not open %s", image_path)
        raise OSError(e)
    return wsi


def save_image(image_path: Path, image: np.ndarray) -> None:
    """Saves a provided image to a given path.

    Args:
        image_path (Path): Path of the image
        image (np.ndarray): Image to save
    """
    try:
        with open(image_path, 'w') as f:
            image.save(f)
    except OSError as e:
        logging.critical("Could not write to %s", image_path)
        raise OSError(e)
