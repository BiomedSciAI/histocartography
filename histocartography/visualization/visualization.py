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
            instance_style (str): Defines how to represent the instances (when available).
                                  Options are 'fill', 'outline', 'fill+outline'. Defaults to 'outline'.
            color (str): Matplotlib named color to (fill) or (outline) the instances. Defaults to 'black'.
            thickness (int): Thickness of the instance outline. Defaults to 1.
            colormap (str): Colormap to use to map labels to colors. Defaults to None.
            alpha (float): Blending of the background image to the instances. Defaults to 0.5.
        """
        super().__init__(**kwargs)
        self.instance_style = instance_style
        self.color = color
        self.thickness = thickness
        self.colormap = colormap
        self.alpha = alpha

    def _process(
        self,
        canvas: np.ndarray,
        instance_map: np.ndarray = None,
        instance_attributes: dict = None,
    ) -> Image:
        """
        Process the image visualization

        Args:
            canvas (np.ndarray): Background on top of which the visualization is drawn.
            instance_map (np.ndarray): Segmentation mask of instances, binary or with individual labels
                                       for each instance. Defaults to None.
            instance_attributes (dict): Dictionary of attributes to be applied to instances.
                                        Defaults to None.

        Returns:
            viz_canvas (Image): Canvas with visualization.
        """

        viz_canvas = self.draw_instances(
            canvas, instance_map, instance_attributes)
        return viz_canvas

    @abstractmethod
    def draw_instances(
            self,
            canvas: np.ndarray,
            instance_map: np.ndarray,
            instance_attributes: dict):
        """
        Abstract method that performs drawing of instances on top of the canvas

        Args:
            canvas (np.ndarray): Background on top of which the visualization is drawn.
            instance_map (np.ndarray): Segmentation mask of instances, brinary or with individual labels for each entity.
            instance_attributes (dict): Dictionary of attributes to be applied to instances.
        """


class InstanceImageVisualization(BaseImageVisualization):
    """
    Instance Image Visualization. Generic instance visualization.
    """

    def draw_instances(
        self,
        canvas: np.ndarray,
        instance_map: np.ndarray = None,
        instance_attributes: dict = None,
    ) -> Image:
        """
        Drawing of instances on top of the canvas

        Args:
            canvas (np.ndarray): Background on top of which the visualization is drawn.
            instance_map (np.ndarray): Segmentation mask of instances, brinary or with individual labels for each entity. Defaults to None.
            instance_attributes (dict): Dictionary of attributes to be applied to instances. Defaults to None.

        Returns:
            viz_canvas (Image): Canvas with visualization.
        """
        if instance_attributes is None:
            instance_attributes = {}
        color = instance_attributes.get(COLOR, self.color)
        colormap = instance_attributes.get(COLORMAP, self.colormap)

        if "outline" in self.instance_style.lower():
            canvas = np.uint8(
                255
                * mark_boundaries(
                    canvas, instance_map, color=name2rgb(color), mode="thick"
                )
            )

        if "fill" in self.instance_style.lower():
            canvas[instance_map > 0] = canvas[instance_map > 0] * \
                (1 - self.alpha)
            if colormap is not None:
                number_of_colors = np.max(instance_map)
                cmap = matplotlib.cm.get_cmap(colormap, number_of_colors)
            else:
                cmap = matplotlib.colors.ListedColormap([color, color])
            colorized_instance_map = instance_map.astype(np.float32) / np.max(
                instance_map
            )
            colorized_instance_map = np.uint8(
                cmap(colorized_instance_map) * 255)[:, :, 0:3]
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
        self,
        instance_visualizer: BaseImageVisualization = None,
        min_max_color_normalize: bool = True,
        **kwargs
    ) -> None:
        """
        Constructor

        Args:
            instance_visualizer (BaseImageVisualization): Instance visualization object. Defaults to None.
            min_max_color_normalize (bool): If the node/edge values, eg importance scores, should be min/max normalized. 
                                            Only relevant if node/edge-level colors are provided. Defaults to True.
        """
        super().__init__(**kwargs)
        self.instance_visualizer = instance_visualizer
        self.min_max_color_normalize = min_max_color_normalize

    def _process(
        self,
        canvas: np.ndarray,
        graph: dgl.DGLGraph,
        instance_map: np.ndarray = None,
        node_attributes: dict = None,
        edge_attributes: dict = None,
        instance_attributes: dict = None,
    ) -> Image:
        """Visualize a graph on an image.

        Args:
            canvas (np.ndarray): Background on top of which the visualization is drawn.
            graph (dgl.DGLGraph): Graph to represent on the canvas.
            instance_map (np.ndarray, optional): Segmentation mask of instances, brinary or with
                                                 individual labels for each entity. Defaults to None.
            node_attributes (dict, optional): Style attribute for the nodes. Defaults to None.
            edge_attributes (dict, optional): Style attribute for the edges. Defaults to None.
            instance_attributes (dict, optional): Dictionary of attributes to be applied to instances.. Defaults to None.

        Returns:
            Image: Visualization output.
        """

        viz_canvas = self.draw_instances(
            canvas, instance_map, instance_attributes)
        draw = ImageDraw.Draw(viz_canvas, "RGBA")

        graph = self.graph_preprocessing(graph)

        self.draw_edges(draw, graph, edge_attributes)
        self.draw_nodes(draw, graph, node_attributes)

        return viz_canvas

    @abstractmethod
    def draw_nodes(
            self,
            draw: ImageDraw,
            graph: dgl.DGLGraph,
            node_attributes: dict):
        """
        Draw nodes on the canvas
        """

    @abstractmethod
    def draw_edges(
            self,
            draw: ImageDraw,
            graph: dgl.DGLGraph,
            edge_attributes: dict):
        """
        Draw edges on the canvas
        """

    @abstractmethod
    def draw_instances(
            self,
            canvas: np.ndarray,
            instance_map: np.ndarray,
            instance_attributes: dict):
        """
        Draw instances on the canvas
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

        Args:
            node_style (str, optional): Style to represent the nodes. Options are "filled",
                                        "outline" or "filled+outline".  Defaults to "outline".
            node_color (str, optional): Node color. Defaults to "yellow".
            node_radius (int, optional): Node radius. Defaults to 5.
            edge_style (str, optional): Edge style. Defaults to "line".
            edge_color (str, optional): Edge color. Defaults to "blue".
            edge_thickness (int, optional): Edge thickness. Defaults to 2.
            colormap (str, optional): Matplotlib colormap. Defaults to "viridis".
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
            self,
            draw: ImageDraw,
            graph: dgl.DGLGraph,
            node_attributes: dict = None):
        """
        Draws the nodes on top of the canvas.
        """

        # extract centroids
        centroids = graph.ndata[CENTROID]

        if node_attributes is None:
            node_attributes = {}

        colors = node_attributes.get(COLOR, [self.node_color])
        if not isinstance(colors[0], str) and self.min_max_color_normalize:
            colors = (colors - np.min(colors))/np.ptp(colors)
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
            self,
            draw: ImageDraw,
            graph: dgl.DGLGraph,
            edge_attributes: dict = None):
        """
        Draws the edges on top of the canvas.
        """
        # extract centroids
        centroids = graph.ndata[CENTROID]
        src, dst = graph.edges()
        edges = [(s.item(), d.item()) for (s, d) in zip(src, dst)]

        if edge_attributes is None:
            edge_attributes = {}

        colors = edge_attributes.get(COLOR, [self.edge_color])
        if not isinstance(colors[0], str) and self.min_max_color_normalize:
            colors = (colors - np.min(colors))/np.ptp(colors)
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
        Draws instances on the canvas.
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
            cell_visualizer (BaseGraphVisualization): Object to use for the cell visualization. Defaults to None.
            tissue_visualizer (BaseGraphVisualization): Object to use for the tissue visualization. Defaults to None.
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
                    alpha=0.10,
                ),
                node_style=None,
                node_color='blue',
                edge_color="blue",
            )
        self.tissue_visualizer = tissue_visualizer

    def _process(
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
            canvas (np.ndarray): Image on which to draw the hierarchical graph.
            cell_graph (dgl.DGLGraph): Cell graph.
            tissue_graph (dgl.DGLGraph): Tissue graph.
            cell_instance_map (np.ndarray, optional): Instance map for the cell graph. Defaults to None.
            cell_node_attributes (dict, optional): Specific attributes of the cell nodes. Defaults to None.
            cell_edge_attributes (dict, optional): Specific attributes of the cell to cell edges. Defaults to None.
            cell_instance_attributes (dict, optional): Specific attributes of the cell instance map. Defaults to None.
            tissue_instance_map (np.ndarray, optional): Instance map for the tissue graph. Defaults to None.
            tissue_node_attributes (dict, optional): Specific attributes of the tissue nodes. Defaults to None.
            tissue_edge_attributes (dict, optional): Specific attributes of the tissue to tissue edges. Defaults to None.
            tissue_instance_attributes (dict, optional): Specific attributes of the tissue instance map. Defaults to None.

        Returns:
            Image: Visualization output.
        """

        cell_centroids = cell_graph.ndata[CENTROID]

        if tissue_instance_map is not None:
            if cell_node_attributes is None:
                cell_node_attributes = {}
                cell_node_attributes["color"] = []
                for centroid in cell_centroids:
                    x = int(centroid[1])
                    y = int(centroid[0])
                    cell_node_attributes["color"].append(
                        tissue_instance_map[x, y])

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
