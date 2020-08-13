from PIL import ImageDraw, Image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
import numpy as np
import dgl
from matplotlib import cm
# from dgl import BatchedDGLGraph
from dgl import DGLGraph
import networkx as nx 
import torch 
from PIL import ImageFilter
from PIL import Image

from histocartography.utils.io import show_image, save_image, complete_path, check_for_dir
from histocartography.utils.draw_utils import draw_ellipse, draw_line, draw_poly, draw_large_circle
from histocartography.ml.layers.constants import CENTROID


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


class GraphVisualization:

    def __init__(self, show_centroid=True, show_edges=True, save=False, save_path='../../data/graphs', verbose=False):
        if verbose:
            print('Initialize graph visualizer')
        self.show_centroid = show_centroid
        self.show_edges = show_edges
        self.save_path = save_path
        self.verbose = verbose
        self.save = save

    def __call__(self, show_cg, show_sg, show_superpx, data, node_importance=None):

        image = data[-2].copy()
        image_name = data[-1]
        try:
            seg_map = data[-1]
            seg_map = None
        except:
            seg_map = None

        if show_sg:
            canvas = image.copy()
            draw = ImageDraw.Draw(canvas, 'RGBA')
            if show_superpx:
                superpx_map = data[2][index] if show_cg else data[1][index]
                self.draw_superpx(superpx_map, draw)
            superpx_graph = dgl.unbatch(data[1])[index] if show_cg else dgl.unbatch(data[0])[index]

            # get centroids and edges
            cent_sp, edges_sp = self._get_centroid_and_edges(superpx_graph)
            self.draw_centroid(cent_sp, draw, (255, 0, 0))
            self.draw_edges(cent_sp, edges_sp, draw, (255, 255, 0), 2)

            if self.save:
                check_for_dir(self.save_path)
                save_image(complete_path(self.save_path, image_name + '_tissue_graph.png'), canvas)

            return canvas

        if show_cg:
            canvas = image.copy()
            draw = ImageDraw.Draw(canvas, 'RGBA')
            
            # if isinstance(data[0], BatchedDGLGraph):
            #     cell_graph = dgl.unbatch(data[0])[index]
            # else:
            #     cell_graph = data[0]

            cell_graph = data[0]

            # get centroids and edges
            cent_cg, edges_cg = self._get_centroid_and_edges(cell_graph)

            # @TODO: hack alert store the centroid and the edges
            self.centroid_cg = cent_cg
            self.edges_cg = edges_cg
            
            # draw centroids
            if self.show_centroid:
                self.draw_centroid(cent_cg, draw, (255, 0, 0))
            
            if seg_map is not None:
                seg_map = seg_map.squeeze()
                mask = Image.new('RGBA', canvas.size, (0, 255, 0, 255))
                alpha = ((seg_map != 0) * 255).astype(np.uint8).squeeze()
                alpha = Image.fromarray(alpha).convert('L')
                # alpha = alpha.filter(ImageFilter.MinFilter(21))
                alpha = alpha.filter(ImageFilter.FIND_EDGES)
                mask.putalpha(alpha)
                canvas.paste(mask, (0, 0), mask)
            
            if self.show_edges:
                self.draw_edges(cent_cg, edges_cg, draw, (255, 255, 0), 2)
            
            if self.save:
                check_for_dir(self.save_path)
                save_image(complete_path(self.save_path, image_name + '_cell_graph.png'), canvas)

            return canvas

    @staticmethod
    def draw_centroid(centroids, draw_bd, fill):
        for centroid in centroids:
            centroid = [centroid[0].item(), centroid[1].item()]
            draw_ellipse(centroid, draw_bd, fill)

    @staticmethod
    def draw_edges(centroids, edges, draw_bd, fill, width):
        for edge in edges:
            src_centroid = [centroids[edge[0]][0].item(), centroids[edge[0]][1].item()]
            dst_centroid = [centroids[edge[1]][0].item(), centroids[edge[1]][1].item()]
            draw_line(src_centroid, dst_centroid, draw_bd, fill, width)

    @staticmethod
    def draw_superpx(mask, draw_bd):
        px_list = list(np.unique(mask))
        for idx, px_id in enumerate(px_list):
            rows, columns = np.where(mask == px_id)
            list_coord = np.asarray([[columns[i].item(), rows[i].item()] for i in range(len(rows))])
            xy = list_coord.flatten().tolist()
            draw_poly(xy, draw_bd)

    @staticmethod
    def _get_centroid_and_edges(graph):
        if isinstance(graph, DGLGraph):
            centroids = graph.ndata[CENTROID]
            src, dst = graph.edges()
            edges = [(s.item(), d.item()) for (s, d) in zip(src, dst)]
        else:
            centroids = nx.get_node_attributes(graph, 'centroid')
            centroids = torch.stack([val for key, val in centroids.items()])
            edges = graph.edges()

        return centroids, edges


def agg_and_plot_interpretation(meta_data, save_path, image_name):

    plt.figure(1)
    plt.title('Explanation Visualization')
    gs = gridspec.GridSpec(6, 3)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 5}

    matplotlib.rc('font', **font)
    label_set = meta_data['output']['label_set']
    y_pos = np.arange(len(label_set))

    # 1. load image of the original graph
    plt.subplot(gs[0:5, 0])
    original_image_name = image_name.replace('_explanation', '')
    original = plt.imread(complete_path(save_path, original_image_name + '_cell_graph.png'))
    plt.imshow(original)
    plt.axis('off')
    plt.title('Original cell graph | Label: ' + meta_data['output']['label'])

    # 2. generate histogram of probability for the original prediction
    plt.subplot(gs[-1, 0])
    probs = list(meta_data['output']['original']['logits'])
    probs = [100 * x for x in probs]

    plt.bar(y_pos, probs, align='center')
    plt.xticks(y_pos, label_set)
    plt.title('Original probability predictions (%)')

    # 3. load image of the explanation graph
    plt.subplot(gs[0:5, 1])
    explanation = plt.imread(complete_path(save_path, image_name + '_explanation_cell_graph.png'))
    plt.imshow(explanation)
    plt.axis('off')
    plt.title('Explanation cell graph.')

    # 4. generate histogram of probability for the explanation
    plt.subplot(gs[-1, 1])
    probs = list(meta_data['output']['explanation']['logits'])
    probs = [100 * x for x in probs]
    plt.bar(y_pos, probs, align='center')
    plt.xticks(y_pos, label_set)
    plt.title('Explanation probability predictions (%)')

    # 5. load image of the explanation graph
    plt.subplot(gs[0:5, 2])
    print('image name', image_name)
    random = plt.imread(complete_path(save_path, image_name + '_random_cell_graph.png'))
    plt.imshow(random)
    plt.axis('off')
    plt.title('Random cell graph.')

    # 6. generate histogram of probability for the explanation
    plt.subplot(gs[-1, 2])
    probs = list(meta_data['output']['random']['res'][0]['logits'])
    probs = [100 * x for x in probs]
    plt.bar(y_pos, probs, align='center')
    plt.xticks(y_pos, label_set)
    plt.title('Random probability predictions (%)')

    # 7. save the image
    plt.savefig(complete_path(save_path, image_name + '_summary_.pdf'), format='pdf', dpi=1200)
