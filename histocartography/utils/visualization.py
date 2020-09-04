from PIL import ImageDraw, Image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
import numpy as np
import dgl
from matplotlib import cm
from sklearn.manifold import TSNE
# from dgl import BatchedDGLGraph
from dgl import DGLGraph
import networkx as nx 
import torch 
from PIL import ImageFilter
from PIL import Image
import bisect 
from skimage.measure import regionprops

from histocartography.utils.io import show_image, save_image, complete_path, check_for_dir
from histocartography.utils.draw_utils import draw_ellipse, draw_line, draw_poly, draw_large_circle, rgb
from histocartography.ml.layers.constants import CENTROID
from histocartography.utils.vector import create_buckets


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


class GraphVisualization:

    def __init__(self, show_centroid=True, show_edges=False, save=False, save_path='../../data/graphs', verbose=False):
        if verbose:
            print('Initialize graph visualizer')
        self.show_centroid = show_centroid
        self.show_edges = show_edges
        self.save_path = save_path
        self.verbose = verbose
        self.save = save

    def __call__(self, show_cg, show_sg, data, node_importance=None):

        graph = data[0]
        image = data[1].copy()
        image_name = data[2]

        canvas = image.copy()
        draw = ImageDraw.Draw(canvas, 'RGBA')

        if show_sg:
            # draw superpixels 
            self.draw_superpx(graph['instance_map'], draw, (255, 0, 0), node_importance)

            if self.save:
                check_for_dir(self.save_path)
                save_image(complete_path(self.save_path, image_name + '_tissue_graph.png'), canvas)

            return canvas

        if show_cg:
            # get centroids and edges
            if isinstance(graph, dict):
                cent_cg = graph['centroid']
                edges_cg = None
            else:
                cent_cg, edges_cg = self._get_centroid_and_edges(graph)

            # @TODO: hack alert store the centroid and the edges
            self.centroid_cg = cent_cg
            self.edges_cg = edges_cg
            
            # draw centroids
            if self.show_centroid:
                self.draw_centroid(cent_cg, draw, (255, 0, 0), node_importance)
            
            # if seg_map is not None:
            #     seg_map = seg_map.squeeze()
            #     mask = Image.new('RGBA', canvas.size, (0, 255, 0, 255))
            #     alpha = ((seg_map != 0) * 255).astype(np.uint8).squeeze()
            #     alpha = Image.fromarray(alpha).convert('L')
            #     # alpha = alpha.filter(ImageFilter.MinFilter(21))
            #     alpha = alpha.filter(ImageFilter.FIND_EDGES)
            #     mask.putalpha(alpha)
            #     canvas.paste(mask, (0, 0), mask)
            
            if self.show_edges:
                self.draw_edges(cent_cg, edges_cg, draw, (255, 255, 0), 2)
            
            if self.save:
                check_for_dir(self.save_path)
                save_image(complete_path(self.save_path, image_name + '_cell_graph.png'), canvas)

            return canvas

    def draw_centroid(self, centroids, draw_bd, fill, node_importance=None):
        if node_importance is not None:
            buckets, colors, sizes = self._get_buckets(node_importance, with_colors_and_sizes=True)

        for centroid_id, centroid in enumerate(centroids):
            centroid = [centroid[0], centroid[1]]
            if node_importance is not None:
                size = sizes[bisect.bisect(buckets, node_importance[centroid_id]) -1]
                outline = colors[bisect.bisect(buckets, node_importance[centroid_id]) -1]
            else:
                raise NotImplementedError("To implement...")
            draw_ellipse(centroid, draw_bd, fill_col=None, size=size, outline=outline)

    def draw_superpx(self, mask, draw_bd, fill, node_importance=None):

        # 1. buckets/colors according to the node importance 
        if node_importance is not None:
            buckets, colors, _ = self._get_buckets(node_importance, with_colors_and_sizes=True) 

        regions = regionprops(np.array(mask).astype(int))

        if len(regions) == len(node_importance):

            # 2. add superpx one by one to canvas 
            for idx, region in enumerate(regions):
                if node_importance is not None:
                    fill = colors[bisect.bisect(buckets, node_importance[idx]) -1]
                else:
                    fill=None

                xy = region['coords']
                xy[:,[0, 1]] = xy[:,[1, 0]]
                xy = list(xy.flatten())
                draw_poly(xy, draw_bd, fill=fill)
        else:
            print('Warning: Node importance has {} elements while {} regions were detected'.format(len(node_importance), len(regions)))

    def _get_buckets(self, node_importance, with_colors_and_sizes=False):
        buckets = create_buckets(N_BUCKETS, min(node_importance), max(node_importance))

        if with_colors_and_sizes:
            sizes = {}
            colors = {}
            base_size = 5
            size_increment = 3
            for i, x in enumerate(buckets[:-1]):
                sizes[i] =  base_size + i * size_increment 
                colors[i] = rgb(buckets[0], buckets[-1], x, transparency=None)

        buckets[0] = -10e4
        buckets[-1] = 10e4

        if with_colors_and_sizes:
            return buckets, colors, sizes

        return buckets

    def _get_buckets_and_sizes(self, node_importance):
        buckets = create_buckets(N_BUCKETS, min(node_importance), max(node_importance))
        sizes = {}
        base_size = 5
        size_increment = 5
        for i, x in enumerate(buckets[:-1]):
            sizes[i] =  base_size + i * size_increment  #@TODO... # rgb(buckets[0], buckets[-1], x, transparency=None)
        buckets[-1] = 10e4
        return buckets, sizes

    def _get_buckets_and_colors(self, node_importance):
        buckets = create_buckets(N_BUCKETS, min(node_importance), max(node_importance))
        colors = {}
        for i, x in enumerate(buckets[:-1]):
            colors[i] = rgb(buckets[0], buckets[-1], x, transparency=None)
        buckets[-1] = 10e4
        return buckets, colors

    @staticmethod
    def draw_edges(centroids, edges, draw_bd, fill, width):
        for edge in edges:
            src_centroid = [centroids[edge[0]][0].item(), centroids[edge[0]][1].item()]
            dst_centroid = [centroids[edge[1]][0].item(), centroids[edge[1]][1].item()]
            draw_line(src_centroid, dst_centroid, draw_bd, fill, width)


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
