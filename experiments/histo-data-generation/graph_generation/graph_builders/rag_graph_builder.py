from skimage.future import graph
import numpy as np
import pandas as pd
import cv2
from skimage.measure import regionprops

from histocartography.graph_generation.graph_builders.base_graph_builder import BaseGraphBuilder



class RAGGraphBuilder(BaseGraphBuilder):
    """
    Super-pixel Graphs class for graph building.
    """

    def __init__(self, config, cuda=False, verbose=False):
        """
        Superpx Graph Builder constructor.

        Args:
            config: list of required params to build a graph
            cuda: (bool) if cuda is available
            verbose: (bool) verbosity level
        """
        super(RAGGraphBuilder, self).__init__(config, cuda, verbose)

        if verbose:
            print('*** Build Super-pixel graph ***')

        self.config = config
        self.cuda = cuda

    def _build_topology(self, instance_map, graph):
        """
        Build topology for superpx
        """

        instance_map = instance_map.numpy()
        inst_ids = np.sort(pd.unique(np.ravel(instance_map))).astype(int)
        kernel = np.ones((5, 5), np.uint8)
        adjacency = np.zeros(shape=(len(inst_ids), len(inst_ids)))
        for id in inst_ids:
            mask = np.array(instance_map == id, np.uint8)
            dilation = cv2.dilate(mask, kernel, iterations=1)
            boundary = dilation - mask
            mask = boundary * instance_map
            idx = np.array(pd.unique(np.ravel(mask))).astype(int)[1:]  # remove 0
            id -= 1  # because instance_map id starts from 1
            idx -= 1  # because instance_map id starts from 1
            adjacency[id, idx] = 1

        edge_list = np.nonzero(adjacency)
        edge_list = [[s, d] for (s, d) in zip(edge_list[0], edge_list[1])]

        # append edges
        edge_list = ([s for (s, _) in edge_list], [d for (_, d) in edge_list])
        graph.add_edges(list(edge_list[0]), list(edge_list[1]))

