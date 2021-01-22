from PIL import ImageDraw, Image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
import numpy as np
import dgl
from matplotlib import cm
from sklearn.manifold import TSNE
from dgl import DGLGraph
import networkx as nx 
import torch 
from PIL import ImageFilter
from PIL import Image
import bisect 
from skimage.measure import regionprops
from typing import Tuple

from ..utils.io import show_image, save_image, complete_path, check_for_dir
from ..utils.draw_utils import draw_ellipse, draw_line, draw_poly, draw_large_circle, rgb
from ..ml.layers.constants import CENTROID
from ..utils.vector import create_buckets
from ..preprocessing.pipeline import PipelineStep


N_BUCKETS = 10


class tSNE:
    def __init__(self, num_dims=2):
        print('Initialize t-SNE')
        self.tsne_op = TSNE(n_components=num_dims)

    def __call__(self, data):
        data_embedded = self.tsne_op.fit_transform(data)
        return data_embedded


def overlay_mask(img, mask, colormap='jet', alpha=0.7):
    """Overlay a colormapped mask on a background image

    Args:
        img (PIL.Image.Image): background image
        mask (PIL.Image.Image): mask to be overlayed in grayscale
        colormap (str, optional): colormap to be applied on the mask
        alpha (float, optional): transparency of the background image

    Returns:
        PIL.Image.Image: overlayed image
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError('img and mask arguments need to be PIL.Image')

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError('alpha argument is expected to be of type float between 0 and 1')

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, 1:]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img


class GraphVisualization(PipelineStep):

    def __init__(
        self,
        show_centroid: bool =True,
        show_edges: bool =False,
        centroid_outline: Tuple[int, int, int] =(0, 0, 255),
        centroid_fill: Tuple[int, int, int] =(0, 0, 255),
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.show_centroid = show_centroid
        self.show_edges = show_edges
        self.centroid_outline = centroid_outline
        self.centroid_fill = centroid_fill

    def process(
        self,
        image: np.ndarray,
        graph: dgl.DGLGraph,
        node_importance: np.ndarray = None,
        instance_map: np.ndarray = None
        ) -> Image:
        """
        Visualize an image along with a graph. 

        Args:
            image (np.ndarray): Image 
            graph (dgl.DGLGraph): Graph. Must include centroids in ndata. 
            node_importance (np.ndarray): Node importance scores. Default to None. 
            instance_map (np.ndarray): Instance map. Default to None. 

        Return:
            canvas (PIL Image): Image with overlaid graph. 
        """
        image = Image.fromarray(image)
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas, 'RGBA')

        # extract centroids and edges from graph 
        cent_cg, edges_cg = self._get_centroid_and_edges(graph)

        if self.show_edges:
            self.draw_edges(cent_cg, edges_cg, draw, (255, 255, 0), 2)
        
        # draw centroids
        if self.show_centroid:
            self.draw_centroid(cent_cg, draw, (255, 0, 0), node_importance)
        
        if instance_map is not None:
            instance_map = instance_map.squeeze()
            instance_ids = list(np.unique(instance_map))
            mask = Image.new('RGBA', canvas.size, (0, 255, 0, 255))
            alpha = np.zeros(instance_map.shape)
            for instance_id in instance_ids:
                alpha += ((instance_map == instance_id) * 255).astype(np.uint8).squeeze()
                alpha = Image.fromarray(alpha).convert('L')
                # alpha = alpha.filter(ImageFilter.MinFilter(21))
                alpha = alpha.filter(ImageFilter.FIND_EDGES)
                alpha = np.array(alpha)
            alpha = Image.fromarray(alpha).convert('L')
            mask.putalpha(alpha)
            mask.save('../data/test.png')
            canvas.paste(mask, (0, 0), mask)
        
        return canvas

    def draw_centroid(self, centroids, draw_bd, fill, node_importance=None):
        if node_importance is not None:
            buckets, colors = self._get_buckets(node_importance, with_colors=True)            

        for centroid_id, centroid in enumerate(centroids):
            centroid = [centroid[1], centroid[0]]
            if node_importance is not None:
                outline = colors[bisect.bisect(buckets, node_importance[centroid_id]) -1]
                fill_col = outline
            else:
                outline = self.centroid_outline
                fill_col = self.centroid_fill
            draw_ellipse(centroid, draw_bd, fill_col=fill_col, outline=outline)

    @staticmethod
    def _get_buckets(node_importance, with_colors=False):
        buckets = create_buckets(N_BUCKETS, min(node_importance), max(node_importance))

        if with_colors:
            colors = {}
            for i, x in enumerate(buckets[:-1]):
                colors[i] = rgb(buckets[0], buckets[-1], x, transparency=None)

        buckets[0] = -10e4
        buckets[-1] = 10e4

        if with_colors:
            return buckets, colors

        return buckets

    @staticmethod
    def draw_edges(centroids, edges, draw_bd, fill, width):
        for edge in edges:
            src_centroid = [centroids[edge[0]][0].item(), centroids[edge[0]][1].item()]
            dst_centroid = [centroids[edge[1]][0].item(), centroids[edge[1]][1].item()]
            draw_line(src_centroid, dst_centroid, draw_bd, fill, width)

    @staticmethod
    def _get_centroid_and_edges(graph):
        centroids = graph.ndata[CENTROID]
        src, dst = graph.edges()
        edges = [(s.item(), d.item()) for (s, d) in zip(src, dst)]
        return centroids, edges
