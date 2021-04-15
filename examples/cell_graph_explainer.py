"""
Example: Explain a cell graph (CG) prediction using a pretrained CG-GNN
         and a graph explainer: GraphGradCAM.

As used in:
"Quantifying Explainers of Graph Neural Networks in Computational Pathology", Jaume et al, CVPR, 2021.
"""

import os
from glob import glob
from PIL import Image
import yaml
import numpy as np
from tqdm import tqdm
import torch
from dgl.data.utils import load_graphs

from histocartography.utils import set_graph_on_cuda, download_test_data
from histocartography.ml import CellGraphModel
from histocartography.interpretability import GraphGradCAMExplainer
from histocartography.visualization import OverlayGraphVisualization, InstanceImageVisualization


IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
NODE_DIM = 514


def explain_cell_graphs(cell_graph_path, image_path):
    """
    Generate an explanation for all the cell graphs in cell path dir.
    """

    # 1. get cell graph & image paths
    cg_fnames = glob(os.path.join(cell_graph_path, '*.bin'))
    image_fnames = glob(os.path.join(image_path, '*.png'))

    # 2. create model
    config_fname = os.path.join(
        os.path.dirname(__file__),
        'config',
        'cg_bracs_cggnn_3_classes_gin.yml')
    with open(config_fname, 'r') as file:
        config = yaml.load(file)

    model = CellGraphModel(
        gnn_params=config['gnn_params'],
        classification_params=config['classification_params'],
        node_dim=NODE_DIM,
        num_classes=3,
        pretrained=True
    ).to(DEVICE)

    # 3. define the explainer
    explainer = GraphGradCAMExplainer(model=model)

    # 4. define graph visualizer
    visualizer = OverlayGraphVisualization(
        instance_visualizer=InstanceImageVisualization(),
        colormap='jet',
        node_style="fill"
    )

    # 5. process all the images
    for cg_path in tqdm(cg_fnames):

        # a. load the graph
        _, graph_name = os.path.split(cg_path)
        graph, _ = load_graphs(cg_path)
        graph = graph[0]
        graph = set_graph_on_cuda(graph) if IS_CUDA else graph

        # b. load corresponding image
        image_path = [
            x for x in image_fnames if graph_name in x.replace(
                '.png', '.bin')][0]
        _, image_name = os.path.split(image_path)
        image = np.array(Image.open(image_path))

        # c. run explainer
        importance_scores, _ = explainer.process(graph)

        # d. visualize and save the output
        node_attrs = {
            "color": importance_scores
        }
        canvas = visualizer.process(image, graph, node_attributes=node_attrs)
        canvas.save(os.path.join('output', 'explainer', image_name))


if __name__ == "__main__":

    # 1. download pre-computed images/cell_graph
    download_test_data('output')

    # 2. create output directories
    os.makedirs(os.path.join('output', 'explainer'), exist_ok=True)

    # 3. generate tissue graphs
    explain_cell_graphs(
        cell_graph_path=os.path.join('output', 'cell_graphs'),
        image_path=os.path.join('output', 'images')
    )
