from PIL import ImageDraw, Image
import numpy as np
import dgl
from PIL import ImageFilter
from PIL import Image
import bisect 
from typing import Tuple, List 
from skimage.segmentation import mark_boundaries

from ..utils.draw_utils import draw_ellipse, draw_line, rgb
from ..ml.layers.constants import CENTROID
from ..utils.vector import create_buckets
from ..pipeline import PipelineStep


N_BUCKETS = 10


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
        instance_map: np.ndarray = None
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
        draw = ImageDraw.Draw(canvas, 'RGBA')

        # extract centroids and edges from graph 
        cent_cg, edges_cg = self._get_centroid_and_edges(graph)

        if self.show_edges:
            self.draw_edges(cent_cg, edges_cg, draw, (255, 255, 0), 2)
        
        # draw centroids
        if self.show_centroid:
            self.draw_centroid(cent_cg, draw, (255, 0, 0), node_importance)
        
        if instance_map is not None:
            if np.sum(instance_map == 0) == 0:  # we have dense segmentation masks
                canvas = self._draw_dense_instance_map(image, instance_map, node_importance)
            else:
                instance_map = instance_map.squeeze()
                mask = Image.new('RGBA', canvas.size, (0, 255, 0, 255))
                alpha_ch = ((instance_map != 0) * 255).astype(np.uint8).squeeze()
                alpha_ch = Image.fromarray(alpha_ch).convert('L')
                alpha_ch = alpha_ch.filter(ImageFilter.FIND_EDGES)
                mask.putalpha(alpha_ch)
                canvas.paste(mask, (0, 0), mask)
        
        return canvas

    def _draw_dense_instance_map(
            self,
            image: np.ndarray,
            mask: np.ndarray,
            node_importance: np.ndarray = None
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
            id_to_imp_map = {k: v for k, v in zip(list(np.unique(mask)), list(node_importance))}
            mask = np.vectorize(mp1)(mask, id_to_imp_map)

            # 3. map importance score to color 
            imp_to_color_map = {imp:np.array(colors[bisect.bisect(buckets, imp) - 1]) for imp in node_importance}
            def mp2(x):
                return imp_to_color_map[x]
            mask = np.vectorize(mp2, signature='()->(n)')(mask)
            canvas = Image.fromarray((self.alpha * np.asarray(image) + (1 - self.alpha) * mask).astype(np.uint8))
        else:
            canvas = Image.fromarray((mark_boundaries(np.array(image), mask)*255).astype(np.uint8))
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
                outline = colors[bisect.bisect(buckets, node_importance[centroid_id]) -1]
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
            width: int = 2
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
