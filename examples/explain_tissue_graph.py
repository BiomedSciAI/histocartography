import numpy as np
import torch
import os 
import h5py
from dgl.data.utils import load_graphs
import argparse
from tqdm import tqdm 
import h5py

from histocartography.interpretability.saliency_explainer.graph_gradcam_explainer import GraphGradCAMExplainer
from histocartography.interpretability.saliency_explainer.graph_gradcampp_explainer import GraphGradCAMPPExplainer
from histocartography.utils.graph import set_graph_on_cuda
from histocartography.utils.io import load_image, h5_to_tensor
from histocartography.preprocessing.superpixel import ColorMergedSuperpixelExtractor
from histocartography.visualisation.graph_visualization import GraphVisualization

BASE_S3 = 's3://mlflow/'
IS_CUDA = torch.cuda.is_available()


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
        '-tg',
        '--tg_path',
        type=str,
        help='path to preprocessed tissue graphs.',
        required=True
    )
    parser.add_argument(
        '-imap',
        '--instance_map_path',
        type=str,
        help='path to preprocessed instance maps.',
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


class TGExplainer:
    """TGExplainer class."""

    def __init__(self, out_path, verbose=True):
        self.out_path = out_path
        self.verbose = verbose
        os.makedirs(os.path.join(out_path, 'explainer_viz'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'explainers'), exist_ok=True)

        self.explainer = GraphGradCAMPPExplainer(
            model_path=BASE_S3 + 'a47b5c2fdf4d49b388e67b63c3c7a8fc/artifacts/model_best_val_weighted_f1_score_0'  
        )

        self.visualiser = GraphVisualization(
            show_centroid=False,
            show_edges=False
        )

    def process(self, image_path, tg_path, imap_path, fnames_path):
        """
        Graph GradCAM to explain a Tissue Graph.
        """

        with open(fnames_path, 'r') as f:
            image_fnames = [line.strip() for line in f]
        tg_fnames = [f.replace('.png', '_tg.bin') for f in image_fnames]
        imap_fnames = [f.replace('.png', '_tg_instance_map.h5') for f in image_fnames]

        for image_name, tg_name, imap_name in tqdm(zip(image_fnames, tg_fnames, imap_fnames)):

            if self.verbose:
                print('*** Testing tissue graph explainer {}'.format(tg_name))

            # 1. load tissue graph, image and instance map
            image = np.array(load_image(os.path.join(image_path, image_name)))

            with h5py.File(os.path.join(imap_path, imap_name), 'r') as f:
                instance_map = h5_to_tensor(f['instance_map'], 'cpu').numpy()
                f.close()

            tissue_graph, _ = load_graphs(os.path.join(tg_path, tg_name))
            tissue_graph = tissue_graph[0]
            tissue_graph.ndata['feat'] = torch.cat(
                (tissue_graph.ndata['feat'].float(),
                (tissue_graph.ndata['centroid'] / torch.FloatTensor(image.shape[:-1])).float()),
                dim=1
            )

            if IS_CUDA:
                tissue_graph = set_graph_on_cuda(tissue_graph)

            # 2. run the explainer
            importance_scores, logits = self.explainer.process(tissue_graph)

            # 3. print output
            if self.verbose:
                print('Number of nodes:', tissue_graph.number_of_nodes())
                print('Number of edges:', tissue_graph.number_of_edges())
                print('Node features:', tissue_graph.ndata['feat'].shape)
                print('Node centroids:', tissue_graph.ndata['centroid'].shape)
                print('Logits:', logits.shape)
                print('Prediction: [Normal, Benign, Atypical, DCIS, Invasive]', logits.squeeze())
                print('Importance scores:', importance_scores)

            # 4. save as h5 file
            with h5py.File(os.path.join(self.out_path, 'explainers', image_name.replace('.png', '_tg_importance.h5')), 'w') as hf:
                hf.create_dataset("importance",  data=importance_scores)

            # 6. visualize the tissue graph
            out = self.visualiser.process(
                image=image,
                graph=tissue_graph,
                node_importance=importance_scores.squeeze(),
                instance_map=instance_map
            )
            tg_fname = image_name.replace('.png', '_tg.png')
            out.save(os.path.join(self.out_path, 'explainer_viz', tg_fname))


if __name__ == "__main__":
    args = parse_arguments()
    explainer = TGExplainer(args.out_path)
    explainer.process(
        args.image_path,
        args.tg_path,
        args.instance_map_path,
        args.fnames_path
    )

