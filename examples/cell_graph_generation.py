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

from histocartography.preprocessing import NucleiExtractor, DeepFeatureExtractor, KNNGraphBuilder
from histocartography.visualization import InstanceImageVisualization
from histocartography.utils.io import load_image

from sklearn.preprocessing import binarize
from sklearn.neighbors import kneighbors_graph

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



class CellGraphBuilder:
    """CellGraphBuilder."""

    def __init__(self, out_path, verbose=True, viz=True):
        self.out_path = out_path
        self.verbose = verbose
        self.viz = viz
        os.makedirs(os.path.join(out_path, 'cell_graphs'), exist_ok=True)

        self.nuclei_detector = NucleiExtractor()
        self.feature_extractor = DeepFeatureExtractor(architecture='resnet34', patch_size=72)
        self.cell_graph_builder = KNNGraphBuilder(thresh=50, add_loc_feats=True)

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
            nuclei_map, nuclei_centroids = self.nuclei_detector.process(image)

            # 3. super pixel feature extraction 
            features = self.feature_extractor.process(image, nuclei_map)

            # 4. build the tissue graph
            cell_graph = self.cell_graph_builder.process(
                structure=nuclei_centroids,
                features=features,
                image_size=(image.shape[1], image.shape[0])
            )

            # 5. print graph properties
            if self.verbose:
                print('Number of nodes:', cell_graph.number_of_nodes())
                print('Number of edges:', cell_graph.number_of_edges())
                print('Node features:', cell_graph.ndata['feat'].shape)
                print('Node centroids:', cell_graph.ndata['centroid'].shape)

            # 6. save DGL graph
            cg_fname = image_name.replace('.png', '.bin')
            save_graphs(os.path.join(self.out_path, 'cell_graphs', cg_fname), [cell_graph])

            # 7. visualize the graph 
            if self.viz:
                out = self.visualiser.process(image, instance_map=nuclei_map)
                tg_fname = image_name.replace('.png', '_cg.png')
                out.save(os.path.join(self.out_path, 'visualization', tg_fname))


if __name__ == "__main__":
    args = parse_arguments()
    graph_builder = CellGraphBuilder(args.out_path)
    graph_builder.process(args.image_path, args.fnames_path)
