import bisect
from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import cycle
from typing import List, Tuple

import dgl
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from skimage.segmentation import mark_boundaries

from ..ml.layers.constants import CENTROID
from ..pipeline import PipelineStep
from ..utils.draw_utils import (
    draw_circle,
    draw_ellipse,
    draw_line,
    rgb,
    map_value_to_color,
)
from ..utils.vector import create_buckets

N_BUCKETS = 10
COLOR = "color"
RADIUS = "radius"
THICKNESS = "thickness"
COLORMAP = "colormap"


class BaseVisualization(PipelineStep):
    """
    Base visualization class
    """

    def __init__(self, **kwargs) -> None:
        """
        Base visualization class
        """
        super().__init__(**kwargs)

    def process(
        self,
        canvas: np.ndarray,
        graph: dgl.DGLGraph,
        instance_map: np.ndarray = None,
        node_attributes: dict = None,
        edge_attributes: dict = None,
        instance_attributes: dict = None,
    ) -> Image:

        viz_canvas = self.draw_instances(canvas, instance_map, instance_attributes)
        draw = ImageDraw.Draw(viz_canvas, "RGBA")

        graph = self.graph_preprocessing(graph)

        self.draw_edges(draw, graph, edge_attributes)
        self.draw_nodes(draw, graph, node_attributes)

        return viz_canvas

    @abstractmethod
    def draw_nodes(self, draw: ImageDraw, graph: dgl.DGLGraph, node_attributes: dict):
        """
        draw nodes on the canvas
        """

    @abstractmethod
    def draw_edges(self, draw: ImageDraw, graph: dgl.DGLGraph, edge_attributes: dict):
        """
        draw nodes on the canvas
        """

    @abstractmethod
    def draw_instances(
        self, canvas: np.ndarray, instance_map: np.ndarray, instance_attributes: dict
    ):
        """
        draw edges on the canvas
        """

    @abstractmethod
    def graph_preprocessing(self, graph: dgl.DGLGraph):
        """
        preprocesses the graph (e.g., to reorganize spatially)
        """


class OverlayGraphVisualization(BaseVisualization):
    def __init__(
        self,
        node_style: str = "outline",
        node_color: str = "yellow",
        node_radius: int = 5,
        edge_style: str = "line",
        edge_color: str = "blue",
        edge_thickness: int = 2,
        colormap="viridis",
        show_colormap=False,
        **kwargs
    ) -> None:
        """
        Overlay graph visualization class. It overlays a graph drawn with
        PIL on top of an image canvas. Nodes outside of the canvas support will
        be ignored.

        Args :
            node_style: str = "outline" or "fill",
            node_color: str = "yellow",
            node_radius: int = 5,
            edge_style: str = "line",
            edge_color: str = "blue",
            edge_thickness: int = 2,
            colormap="viridis",
            show_colormap=False,

        """
        super().__init__(**kwargs)
        self.node_style = node_style
        self.node_color = node_color
        self.node_radius = node_radius
        self.edge_style = edge_style
        self.edge_color = edge_color
        self.edge_thickness = edge_thickness
        self.colormap = colormap

    def graph_preprocessing(self, graph: dgl.DGLGraph):
        return graph

    def draw_nodes(
        self, draw: ImageDraw, graph: dgl.DGLGraph, node_attributes: dict = None
    ):
        """
        Draws the nodes on top of the canvas.
        Args:

            draw : ImageDraw canvas
            graph: dgl.DGLGraph with the information to be added
            node_attributes: dict with any the following keywords ('color', 'radius', 'colormap')

            'color': sting name of the color for all nodes, an iterable of color_values to map with using a Matplotlib 'colormap'
        """

        # extract centroids
        centroids = graph.ndata[CENTROID]

        if node_attributes is None:
            node_attributes = {}

        colors = node_attributes.get(COLOR, [self.node_color])
        radii = node_attributes.get(RADIUS, [self.node_radius])
        colormap = node_attributes.get(COLORMAP, self.colormap)
        thicknesses = node_attributes.get(THICKNESS, [2])
        if not isinstance(colors, Iterable):
            colors = [colors]
        if not isinstance(radii, Iterable):
            radii = [radii]
        if not isinstance(thicknesses, Iterable):
            thicknesses = [thicknesses]

        radius = cycle(radii)
        color = cycle(colors)
        thickness = cycle(thicknesses)

        if self.node_style == "outline":
            for centroid in centroids:
                draw_circle(
                    centroid,
                    draw,
                    outline_color=map_value_to_color(next(color), colormap),
                    fill_color=None,
                    radius=next(radius),
                    width=next(thickness),
                )

        if self.node_style == "fill":
            for centroid in centroids:
                draw_circle(
                    centroid,
                    draw,
                    outline_color=None,
                    fill_color=map_value_to_color(next(color), colormap),
                    radius=next(radius),
                    width=next(thickness),
                )

    def draw_edges(
        self, draw: ImageDraw, graph: dgl.DGLGraph, edge_attributes: dict = None
    ):
        """
        Draws the nodes on top of the canvas.
        Args:

            draw : ImageDraw canvas
            graph: dgl.DGLGraph with the information to be added
            edge_attributes: dict with any the following keywords ('color', 'thickness', 'colormap')

            'color': sting name of the color for all edges, an iterable of color_values to map with using a Matplotlib 'colormap'
        """
        # extract centroids
        centroids = graph.ndata[CENTROID]
        src, dst = graph.edges()
        edges = [(s.item(), d.item()) for (s, d) in zip(src, dst)]

        if edge_attributes is None:
            edge_attributes = {}

        colors = edge_attributes.get(COLOR, [self.edge_color])
        thicknesses = edge_attributes.get(THICKNESS, self.edge_thickness)
        colormap = edge_attributes.get(COLORMAP, self.colormap)
        if not isinstance(colors, Iterable):
            colors = [colors]
        if not isinstance(thicknesses, Iterable):
            thicknesses = [thicknesses]

        color = cycle(colors)
        thickness = cycle(thicknesses)

        if self.edge_style is not None:
            for edge in edges:
                src_centroid = [
                    centroids[edge[0]][1].item(),
                    centroids[edge[0]][0].item(),
                ]
                dst_centroid = [
                    centroids[edge[1]][1].item(),
                    centroids[edge[1]][0].item(),
                ]
                draw_line(
                    src_centroid,
                    dst_centroid,
                    draw,
                    fill_col=map_value_to_color(next(color), colormap),
                    line_wid=next(thickness),
                )

    def draw_instances(
        self,
        canvas: np.ndarray,
        instance_map: np.ndarray = None,
        instance_attributes: dict = None,
    ):
        """
        Draws the canvas of the image using the instance map.
        Args:

            canvas: np.ndarray,
            instance_map: np.ndarray = None,
            instance_attributes: dict = None,
        """

        if instance_map is not None:
            canvas = 255*mark_boundaries(canvas, instance_map, color=(
                0, 0, 0), mode='thick')
            
        image = Image.fromarray(canvas.astype(np.uint8))
        viz_canvas = image.copy()
        
        return viz_canvas


class GraphVisualization(PipelineStep):
    def __init__(
        self,
        show_centroid: bool = True,
        show_edges: bool = False,
        centroid_outline: Tuple[int, int, int] = (0, 0, 255),
        centroid_fill: Tuple[int, int, int] = (0, 0, 255),
        alpha: float = 0.3,
        **kwargs
    ) -> None:
        """
        GraphVisualization constructor

        Args:
            show_centroid (bool, Optional): If a circle around each centroid is drawn. Default to True.
            show_edges (bool, Optional): if the edges are drawn. Default to False.
            centroid_outline (Tuple[int, int, int], Optional): Centroid outline color. Default to (0, 0, 255),
            centroid_fill (Tuple[int, int, int], Optional): Centroid fill color.  Default to (0, 0, 255),
            alpha (float, Optional): Transparency of the overlaid segmentation mask. Default to 0.3.
        """
        super().__init__(**kwargs)
        self.show_centroid = show_centroid
        self.show_edges = show_edges
        self.centroid_outline = centroid_outline
        self.centroid_fill = centroid_fill
        self.alpha = alpha

    def process(
        self,
        image: np.ndarray,
        graph: dgl.DGLGraph,
        node_importance: np.ndarray = None,
        instance_map: np.ndarray = None,
    ) -> Image:
        """
        Visualize an image along with a graph.

        Args:
            image (np.ndarray): Image
            graph (dgl.DGLGraph): Graph. Must include centroids in ndata.
            node_importance (np.ndarray): Node importance scores. Default to None.
            instance_map (np.ndarray): Instance map. Default to None.

        Returns:
            canvas (PIL Image): Image with overlaid graph.
        """
        image = Image.fromarray(image)
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas, "RGBA")

        # extract centroids and edges from graph
        cent_cg, edges_cg = self._get_centroid_and_edges(graph)

        if self.show_edges:
            self.draw_edges(cent_cg, edges_cg, draw, (255, 255, 0), 2)

        # draw centroids
        if self.show_centroid:
            self.draw_centroid(cent_cg, draw, (255, 0, 0), node_importance)

        if instance_map is not None:
            if np.sum(instance_map == 0) == 0:  # we have dense segmentation masks
                canvas = self._draw_dense_instance_map(
                    image, instance_map, node_importance
                )
            else:
                instance_map = instance_map.squeeze()
                mask = Image.new("RGBA", canvas.size, (0, 255, 0, 255))
                alpha_ch = ((instance_map != 0) * 255).astype(np.uint8).squeeze()
                alpha_ch = Image.fromarray(alpha_ch).convert("L")
                alpha_ch = alpha_ch.filter(ImageFilter.FIND_EDGES)
                mask.putalpha(alpha_ch)
                canvas.paste(mask, (0, 0), mask)

        return canvas

    def _draw_dense_instance_map(
        self, image: np.ndarray, mask: np.ndarray, node_importance: np.ndarray = None
    ):
        """
        Draw a dense instance map on an image.

        Args:
            image (np.ndarray): Image
            mask (np.ndarray): Instance map. Default to None.
            node_importance (np.ndarray): Node importance scores. Default to None.

        Returns:
            canvas (PIL Image): Image with overlaid instance map.
        """
        if node_importance is not None:
            # 1. get buckets and colors
            buckets, colors = self._get_buckets(node_importance, with_colors=True)

            # 2. map superpixel id to importance score
            def mp1(entry, mapper):
                return mapper[entry]

            id_to_imp_map = {
                k: v for k, v in zip(list(np.unique(mask)), list(node_importance))
            }
            mask = np.vectorize(mp1)(mask, id_to_imp_map)

            # 3. map importance score to color
            imp_to_color_map = {
                imp: np.array(colors[bisect.bisect(buckets, imp) - 1])
                for imp in node_importance
            }

            def mp2(x):
                return imp_to_color_map[x]

            mask = np.vectorize(mp2, signature="()->(n)")(mask)
            canvas = Image.fromarray(
                (self.alpha * np.asarray(image) + (1 - self.alpha) * mask).astype(
                    np.uint8
                )
            )
        else:
            canvas = Image.fromarray(
                (mark_boundaries(np.array(image), mask) * 255).astype(np.uint8)
            )
        return canvas

    def draw_centroid(self, centroids, draw_bd, fill, node_importance=None):
        """
        Draw centroids on draw_db.
            Args:
                centroids (List): Centroid location.
                draw_bd (ImageDraw.Draw): Drawing tool.
                fill (Tuple[int, int, int], Optional): Default to (255, 255, 0).
                width (int, Optional): Edge stroke width. Default to 2.
                node_importance (np.ndarray, Optional): Node importance scores.
        """
        if node_importance is not None:
            buckets, colors = self._get_buckets(node_importance, with_colors=True)
        for centroid_id, centroid in enumerate(centroids):
            centroid = [centroid[1], centroid[0]]
            if node_importance is not None:
                outline = colors[
                    bisect.bisect(buckets, node_importance[centroid_id]) - 1
                ]
                fill_col = outline
            else:
                outline = self.centroid_outline
                fill_col = self.centroid_fill
            draw_ellipse(centroid, draw_bd, fill_col=fill_col, outline=outline)

    @staticmethod
    def _get_buckets(node_importance, with_colors=False):
        """
        Quantize the node importance scores into N_BUCKETS bins. Additionaly
        associate a color to each bin (low: blue, high: red)

        Args:
            node_importance (np.ndarray): Node importance scores (one per node).
            with_colors (bool, Optional): If we build colors for each bucket.

        Returns:
            buckets (List): Quantized node importances
            colors (Dict): RGB colors asscoiated to each bucket.
        """
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
    def draw_edges(
        centroids: List,
        edges: List,
        draw_bd: ImageDraw.Draw,
        fill: Tuple[int, int, int] = (255, 255, 0),
        width: int = 2,
    ):
        """
        Draw edges on draw_bd
            Args:
                centroids (List): Centroid location.
                edges (List): Edge list.
                draw_bd (ImageDraw.Draw): Drawing tool.
                fill (Tuple[int, int, int], Optional): Default to (255, 255, 0).
                width (int, Optional): Edge stroke width. Default to 2.
        """
        for edge in edges:
            src_centroid = [centroids[edge[0]][1].item(), centroids[edge[0]][0].item()]
            dst_centroid = [centroids[edge[1]][1].item(), centroids[edge[1]][0].item()]
            draw_line(src_centroid, dst_centroid, draw_bd, fill, width)

    @staticmethod
    def _get_centroid_and_edges(graph: dgl.DGLGraph):
        """
        Extract the centroid locations and edges from a DGL graph.

        Args:
            graph (dgl.DGLGraph): A graph

        Returns:
            centroids (torch.FloatTensor): Centroids (N x 2).
            edges (list[Tuple]): Edges of the graph.
        """
        centroids = graph.ndata[CENTROID]
        src, dst = graph.edges()
        edges = [(s.item(), d.item()) for (s, d) in zip(src, dst)]
        return centroids, edges
