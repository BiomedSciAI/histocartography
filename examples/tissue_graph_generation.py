"""Build tissue graphs from a list of images"""
import numpy as np
import cv2 
import torch 
import yaml
import os 
from dgl.data.utils import save_graphs
import argparse
from tqdm import tqdm  
import h5py

from histocartography.preprocessing import ColorMergedSuperpixelExtractor, DeepFeatureExtractor, RAGGraphBuilder
from histocartography.visualization import InstanceImageVisualization
from histocartography.utils.io import load_image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image_path',
        type=str,
        help='path to the images.',
        required=True
    )
    parser.add_argument(
        '-f',
        '--fnames_path',
        type=str,
        help='path to file with list of images to process.',
        required=True
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help='path to save the output.',
        required=True
    )
    return parser.parse_args()



class TissueGraphBuilder:
    """TissueGraphBuilder."""

    def __init__(self, out_path, verbose=True, viz=True):
        self.out_path = out_path
        self.verbose = verbose
        self.viz = viz
        os.makedirs(os.path.join(out_path, 'tissue_graphs'), exist_ok=True)

        self.superpixel_detector = ColorMergedSuperpixelExtractor(
            nr_superpixels=50,
            compactness=20,
            blur_kernel_size=1,
            threshold=0.01,
        )
        self.feature_extractor = DeepFeatureExtractor(architecture='resnet34', patch_size=144)
        self.tissue_graph_builder = RAGGraphBuilder()

        if self.viz:
            self.visualiser = InstanceImageVisualization()
            os.makedirs(os.path.join(out_path, 'visualization'), exist_ok=True)

    def process(self, image_path, fnames_path):
        """
        Process the images listed in the fnames_path file. 
        """

        with open(fnames_path, 'r') as f:
            image_fnames = [line.strip() for line in f]

        for image_name in tqdm(image_fnames):
            print('*** Testing image {}'.format(image_name))

            # 1. load image
            image = np.array(load_image(os.path.join(image_path, image_name)))
    
            # 2. super pixel detection 
            merged_superpixels, superpixels = self.superpixel_detector.process(image)

            # 3. super pixel feature extraction 
            features = self.feature_extractor.process(image, merged_superpixels)

            # 5. build the tissue graph
            tissue_graph = self.tissue_graph_builder.process(
                structure=merged_superpixels,
                features=features,
                image_size=(image.shape[1], image.shape[0])
            )

            # 6. print graph properties
            if self.verbose:
                print('Number of nodes:', tissue_graph.number_of_nodes())
                print('Number of edges:', tissue_graph.number_of_edges())
                print('Node features:', tissue_graph.ndata['feat'].shape)
                print('Node centroids:', tissue_graph.ndata['centroid'].shape)
                print('Image size:', tissue_graph.gdata['image_size'])

            # 7. save DGL graph
            tg_fname = image_name.replace('.png', '.bin')
            save_graphs(os.path.join(self.out_path, 'tissue_graphs', tg_fname), [tissue_graph])

            # 9. visualize the graph 
            if self.viz:
                out = self.visualiser.process(image, instance_map=merged_superpixels)
                tg_fname = image_name.replace('.png', '_tg.png')
                out.save(os.path.join(self.out_path, 'visualization', tg_fname))


if __name__ == "__main__":
    args = parse_arguments()
    graph_builder = TissueGraphBuilder(args.out_path)
    graph_builder.process(args.image_path, args.fnames_path)

