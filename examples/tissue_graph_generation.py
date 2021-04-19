"""
Example: Extract a Tissue Graph from an H&E image.

As used in:
- "Hierarchical Graph Representations in Digital Pathology", Pati et al, 2021.
- "HACT-Net: A Hierarchical Cell-to-Tissue Graph Neural Network for Histopathological Image Classification", Pati et al, MCCAI-W, 2020.
"""

import os
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from dgl.data.utils import save_graphs

from histocartography.utils import download_example_data
from histocartography.preprocessing import (
    ColorMergedSuperpixelExtractor,
    DeepFeatureExtractor,
    RAGGraphBuilder
)
from histocartography.visualization import OverlayGraphVisualization


def generate_tissue_graph(image_path):
    """
    Generate a tissue graph for all the images in image path dir.
    """

    # 1. get image path
    image_fnames = glob(os.path.join(image_path, '*.png'))

    # 2. define superpixel extractor. Here, we query 50 SLIC superpixels,
    # but a superpixel size (in #pixels) can be provided as well in the case
    # where image size vary from one sample to another.
    superpixel_detector = ColorMergedSuperpixelExtractor(
        nr_superpixels=50,
        compactness=20,
        blur_kernel_size=1,
        threshold=0.02,
    )

    # 3. define feature extractor: extract patches of 144x144 pixels
    # resized to 224 to match resnet input size. If the superpixel is larger
    # than 144x144, several patches are extracted and patch embeddings are averaged.
    # Everything is handled internally. Please refer to the implementation for
    # details.
    feature_extractor = DeepFeatureExtractor(
        architecture='resnet34',
        patch_size=144,
        resize_size=224
    )

    # 4. define graph builder
    tissue_graph_builder = RAGGraphBuilder(add_loc_feats=True)

    # 5. define graph visualizer
    visualizer = OverlayGraphVisualization()

    # 6. process all the images
    for image_path in tqdm(image_fnames):

        # a. load image
        _, image_name = os.path.split(image_path)
        image = np.array(Image.open(image_path))

        # b. extract superpixels
        superpixels, _ = superpixel_detector.process(image)

        # c. extract deep features
        features = feature_extractor.process(image, superpixels)

        # d. build a Region Adjacency Graph (RAG)
        graph = tissue_graph_builder.process(superpixels, features)

        # e. save the graph
        fname = image_name.replace('.png', '.bin')
        save_graphs(os.path.join('output', 'tissue_graphs', fname), [graph])

        # f. visualize and save the graph
        canvas = visualizer.process(image, graph, instance_map=superpixels)
        canvas.save(os.path.join('output', 'tissue_graphs_viz', image_name))


if __name__ == "__main__":

    # 1. download dummy images
    download_example_data('output')

    # 2. create output directories
    os.makedirs(os.path.join('output', 'tissue_graphs'), exist_ok=True)
    os.makedirs(os.path.join('output', 'tissue_graphs_viz'), exist_ok=True)

    # 3. generate tissue graphs
    generate_tissue_graph(image_path=os.path.join('output', 'images'))
