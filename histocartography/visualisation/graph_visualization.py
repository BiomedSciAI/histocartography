import bisect
from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import cycle
from typing import List, Tuple

import dgl
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib
from skimage.segmentation import mark_boundaries

from ..ml.layers.constants import CENTROID
from ..pipeline import PipelineStep
from ..utils.draw_utils import (
    draw_circle,
    draw_ellipse,
    draw_line,
    rgb,
    map_value_to_color,
    name2rgb,
)
from ..utils.vector import create_buckets

N_BUCKETS = 10
COLOR = "color"
RADIUS = "radius"
THICKNESS = "thickness"
COLORMAP = "colormap"


class BaseImageVisualization(PipelineStep):
    """
    Base visualization class
    """

    def __init__(
        self,
        instance_style: str = "outline",
        color: str = "black",
        thickness: int = 1,
        colormap: str = None,
        alpha: float = 0.5,
        **kwargs
    ) -> None:
        """
        Base visualization class constructor

        Args:
            instance_style: defines how to represent the instances (when available)('fill', 'outline', 'fill+outline')
            color: matplotlib named color to (fill) or (outline) the instances
            thickness: thickness of the instance outline
            colormap: colormap to use to map labels to colors.
            alpha: blending of the background image to the instances
        """
        super().__init__(**kwargs)
        self.instance_style = instance_style
        self.color = color
        self.thickness = thickness
        self.colormap = colormap
        self.alpha = alpha

    def process(
        self,
        canvas: np.ndarray,
        instance_map: np.ndarray = None,
        instance_attributes: dict = None,
    ) -> Image:

        """
        Process the image visualization

        Args:
            canvas: background on top of which the visualization is drawn
            instance_map: segmentation mask of instances, brinary or with individual labels for each
            instance_attributes: dictionary of attributes to be applied to instances
        """

        viz_canvas = self.draw_instances(canvas, instance_map, instance_attributes)
        draw = ImageDraw.Draw(viz_canvas, "RGBA")

        return viz_canvas

    @abstractmethod
    def draw_instances(
        self, canvas: np.ndarray, instance_map: np.ndarray, instance_attributes: dict
    ):
        """
        Abstract method that performs drawing of instances on top of the canvas

        Args:
            canvas: background on top of which the visualization is drawn
            instance_map: segmentation mask of instances, brinary or with individual labels for each
            instance_attributes: dictionary of attributes to be applied to instances
        """


class InstanceImageVisualization(BaseImageVisualization):
    """
    Instance Image Visualization. Generic instance visualization.
    """

    def __init__(self, **kwargs) -> None:
        """
        Constructor. Args are the same as for the superclass

        Args:
            instance_style: defines how to represent the instances (when available)('fill', 'outline', 'fill+outline')
            color: matplotlib named color to (fill) or (outline) the instances
            thickness: thickness of the instance outline
            colormap: colormap to use to map labels to colors.
            alpha: blending of the background image to the instances
        """
        super().__init__(**kwargs)

    def draw_instances(
        self,
        canvas: np.ndarray,
        instance_map: np.ndarray = None,
        instance_attributes: dict = None,
    ):
        """
        Drawing of instances on top of the canvas

        Args:
            canvas: background on top of which the visualization is drawn
            instance_map: segmentation mask of instances, brinary or with individual labels for each
            instance_attributes: dictionary of attributes to be applied to instances
        """
        if instance_attributes is None:
            instance_attributes = {}
        color = instance_attributes.get(COLOR, self.color)
        thickness = instance_attributes.get(THICKNESS, self.thickness)
        colormap = instance_attributes.get(COLORMAP, self.colormap)

        if "outline" in self.instance_style.lower():
            canvas = np.uint8(
                255
                * mark_boundaries(
                    canvas, instance_map, color=name2rgb(color), mode="thick"
                )
            )

        if "fill" in self.instance_style.lower():
            canvas[instance_map > 0] = canvas[instance_map > 0] * (1 - self.alpha)
            if colormap is not None:
                number_of_colors = np.max(instance_map)
                cmap = matplotlib.cm.get_cmap(colormap, number_of_colors)
            else:
                cmap = matplotlib.colors.ListedColormap([color, color])
            colorized_instance_map = instance_map.astype(np.float32) / np.max(
                instance_map
            )
            colorized_instance_map = np.uint8(cmap(colorized_instance_map) * 255)[
                :, :, 0:3
            ]
            colorized_instance_map[instance_map == 0, :] = 0
            canvas = np.uint8(canvas + colorized_instance_map * self.alpha)
        image = Image.fromarray(canvas)

        viz_canvas = image.copy()

        return viz_canvas


class BaseGraphVisualization(PipelineStep):
    """
    Base visualization class
    """

    def __init__(
        self, instance_visualizer: BaseImageVisualization = None, **kwargs
    ) -> None:
        """
        Constructor

        Args:
            instance_visualizer: object to use for the instance visualization

        """
        super().__init__(**kwargs)
        self.instance_visualizer = instance_visualizer

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


class OverlayGraphVisualization(BaseGraphVisualization):
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
        PIL on top of an image canvas using the provided instance_visualizer.
        Nodes outside of the canvas support willbe ignored.

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

        if self.instance_visualizer is None:
            self.instance_visualizer = InstanceImageVisualization()

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

        number_of_colors = len(set(colors))

        radius = cycle(radii)
        color = cycle(colors)
        thickness = cycle(thicknesses)

        if self.node_style == "outline":
            for centroid in centroids:
                draw_circle(
                    centroid,
                    draw,
                    outline_color=map_value_to_color(
                        next(color), colormap, number_of_colors
                    ),
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
                    fill_color=map_value_to_color(
                        next(color), colormap, number_of_colors
                    ),
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
            canvas = self.instance_visualizer.process(
                canvas,
                instance_map=instance_map,
                instance_attributes=instance_attributes,
            )
        if isinstance(canvas, Image.Image):
            image = canvas
        else:
            image = Image.fromarray(canvas.astype(np.uint8))
        viz_canvas = image.copy()

        return viz_canvas


class HACTVisualization(PipelineStep):
    """
    Hierarchical Cell to Tissue visualization class
    """

    def __init__(
        self,
        cell_visualizer: BaseGraphVisualization = None,
        tissue_visualizer: BaseGraphVisualization = None,
        **kwargs
    ) -> None:
        """
        Constructor

        Args:
            cell_visualizer: object to use for the cell visualization
            tissue_visualizer: object to use for the tissue visualization
        """
        super().__init__(**kwargs)
        if cell_visualizer is None:
            cell_visualizer = OverlayGraphVisualization(
                node_style="fill",
                node_color="black",
                edge_color="black",
                colormap="jet",
            )
        self.cell_visualizer = cell_visualizer
        if tissue_visualizer is None:
            tissue_visualizer = OverlayGraphVisualization(
                instance_visualizer=InstanceImageVisualization(
                    color="blue",
                    instance_style="fill+outline",
                    colormap="jet",
                    alpha=0.20,
                ),
                node_style=None,
                node_color='blue',
                edge_color="black",
            )
        self.tissue_visualizer = tissue_visualizer

    def process(
        self,
        canvas: np.ndarray,
        cell_graph: dgl.DGLGraph,
        tissue_graph: dgl.DGLGraph,
        cell_instance_map: np.ndarray = None,
        cell_node_attributes: dict = None,
        cell_edge_attributes: dict = None,
        cell_instance_attributes: dict = None,
        tissue_instance_map: np.ndarray = None,
        tissue_node_attributes: dict = None,
        tissue_edge_attributes: dict = None,
        tissue_instance_attributes: dict = None,
    ) -> Image:

        """
        Draws the hierarchical graph on top of the canvas.
        Args:

            canvas: image on which to draw the hierarchical graph,
            cell_graph: cell graph,
            tissue_graph: tissue graph,
            cell_instance_map: instance map for the cell graph,
            cell_node_attributes: specific attributes of the cell nodes,
            cell_edge_attributes: specific attributes of the cell to cell edges,
            cell_instance_attributes: dict = specific attributes of the cell instance map,
            tissue_instance_map: instance map for the tissue graph,
            tissue_node_attributes: specific attributes of the tissue nodes,
            tissue_edge_attributes: specific attributes of the tissue to tissue edges,
            tissue_instance_attributes: dict = specific attributes of the tissue instance map,
        """

        cell_centroids = cell_graph.ndata[CENTROID]
        tissue_centroids = tissue_graph.ndata[CENTROID]

        if tissue_instance_map is not None:
            if cell_node_attributes is None:
                cell_node_attributes = {}
                cell_node_attributes["color"] = []
                for centroid in cell_centroids:
                    x = int(centroid[1])
                    y = int(centroid[0])
                    cell_node_attributes["color"].append(tissue_instance_map[x, y])

        cell_canvas = self.cell_visualizer.process(
            canvas,
            graph=cell_graph,
            instance_map=cell_instance_map,
            node_attributes=cell_node_attributes,
            edge_attributes=cell_edge_attributes,
            instance_attributes=cell_instance_attributes,
        )

        viz_canvas = self.tissue_visualizer.process(
            cell_canvas,
            graph=tissue_graph,
            instance_map=tissue_instance_map,
            node_attributes=tissue_node_attributes,
            edge_attributes=tissue_edge_attributes,
            instance_attributes=tissue_instance_attributes,
        )

        return viz_canvas
