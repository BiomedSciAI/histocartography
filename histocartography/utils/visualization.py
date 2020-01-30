from PIL import Image, ImageDraw

from histocartography.utils.io import show_image, save_image, complete_path
from histocartography.ml.layers.constants import CENTROID
import numpy as np
import random


class GraphVisualization:

    def __init__(self, show=False, save=True):
        print('Initialize graph visualizer')
        self.show = show
        self.save = save
        self.save_path = '/Users/frd/Documents/Code/Projects/Experiments/data_dummy_sp/graphs/'

    def __call__(self, image, image_name, model_type, *args):

        draw = ImageDraw.Draw(image, 'RGBA')

        if model_type == 'superpx_graph_model' or model_type == 'multi_level_graph_model' or \
                model_type == 'concat_graph_model':
            superpx_graph = args[0] if model_type == 'superpx_graph_model' else args[1]
            superpx_mask = args[1] if model_type == 'superpx_graph_model' else args[2]

            # get centroids and edges
            cent_sp, edges_sp = self._get_centroid_and_edges(superpx_graph)

            self.draw_superpx(superpx_mask, draw)
            self.draw_superpx_centroid(cent_sp, draw)
            self.draw_superpx_edges(cent_sp, edges_sp, draw)

        if model_type == 'cell_graph_model' or model_type == 'multi_level_graph_model' or \
                model_type == 'concat_graph_model':
            cell_graph = args[0]

            # get centroids and edges
            cent_cg, edges_cg = self._get_centroid_and_edges(cell_graph)

            # draw centroids
            self.draw_cell_centroid(cent_cg, draw)
            self.draw_cell_edges(cent_cg, edges_cg, draw)

        if self.show:
            show_image(image)

        if self.save:
            save_image(image, fname=complete_path(self.save_path, image_name + '.png'))

        return image

    def draw_cell_centroid(self, centroids, draw_bd):
        for centroid in centroids:
            centroid = [centroid[0].item(), centroid[1].item()]
            fill_cg_cent = (255, 0, 0)
            self.draw_ellipse(centroid, draw_bd, fill_cg_cent)

    def draw_superpx_centroid(self, centroids, draw_bd):
        for centroid in centroids:
            centroid = [centroid[1].item(), centroid[0].item()]
            fill_sp_cent = (0, 0, 255)
            self.draw_ellipse(centroid, draw_bd, fill_sp_cent)

    def draw_cell_edges(self, centroids, edges, draw_bd):
        for edge in edges:
            src_centroid = [centroids[edge[0]][0].item(), centroids[edge[0]][1].item()]
            dst_centroid = [centroids[edge[1]][0].item(), centroids[edge[1]][1].item()]
            fill_cg_edge = (255, 0, 0)
            cg_width = 5
            self.draw_line(src_centroid, dst_centroid, draw_bd, fill_cg_edge, cg_width)

    def draw_superpx_edges(self, centroids, edges, draw_bd):
        for edge in edges:
            src_centroid = [centroids[edge[0]][1].item(), centroids[edge[0]][0].item()]
            dst_centroid = [centroids[edge[1]][1].item(), centroids[edge[1]][0].item()]
            fill_sp_edge = (0, 0, 255)
            sp_width = 2
            self.draw_line(src_centroid, dst_centroid, draw_bd, fill_sp_edge, sp_width)

    def draw_superpx(self, mask, draw_bd):
        px_list = list(np.unique(mask))
        for idx, px_id in enumerate(px_list):
            rows, columns = np.where(mask == px_id)
            list_coord = np.asarray([[columns[i].item(), rows[i].item()] for i in range(len(rows))])
            xy = list_coord.flatten().tolist()
            self.draw_poly(xy, draw_bd)

    @staticmethod
    def _get_centroid_and_edges(graph):
        centroids = graph.ndata[CENTROID]
        src, dst = graph.edges()
        edges = [(s.item(), d.item()) for (s, d) in zip(src, dst)]
        return centroids, edges

    @staticmethod
    def draw_ellipse(centroid, draw, fill_col):
        draw.ellipse((centroid[0] - 5, centroid[1] - 5, centroid[0] + 5, centroid[1] + 5),
                     fill=fill_col,
                     outline=(0, 0, 0))

    @staticmethod
    def draw_line(source_centroid, dest_centroid, draw, fill_col, line_wid):
        draw.line((source_centroid[0], source_centroid[1], dest_centroid[0], dest_centroid[1]),
                  fill=fill_col,
                  width=line_wid)

    @staticmethod
    def draw_poly(xy, draw):
        draw.polygon(xy, outline=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 100))
