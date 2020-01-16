import PIL
from PIL import Image, ImageDraw

from histocartography.utils.io import show_image
from histocartography.ml.layers.constants import CENTROID


class GraphVisualization:

    def __init__(self):
        print('Initialize graph visualizer')

    def __call__(self, graph, image):

        height, width = image.size
        centroids = graph.ndata[CENTROID]
        draw = ImageDraw.Draw(image)

        # 1. draw the edges
        src, dst = graph.edges()
        edges = [(s.item(), d.item()) for (s, d) in zip(src, dst)]
        for edge in edges:
            src_centroid = [centroids[edge[0]][0].item() * width, centroids[edge[0]][1].item() * height]
            dst_centroid = [centroids[edge[1]][0].item() * width, centroids[edge[1]][1].item() * height]
            draw.line((src_centroid[0], src_centroid[1], dst_centroid[0], dst_centroid[1]),
                      fill=(255, 255, 0),
                      width=2)

        # 2. draw the nodes
        for centroid in centroids:
            centroid = [centroid[0].item() * width, centroid[1].item() * height]
            draw.ellipse((centroid[0] - 5, centroid[1] - 5, centroid[0] + 5, centroid[1] + 5),
                         fill=(255, 0, 0),
                         outline=(0, 0, 0))

        show_image(image)
