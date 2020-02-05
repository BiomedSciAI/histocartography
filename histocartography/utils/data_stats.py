from collections import defaultdict
import numpy as np


class DataStats:

    def __init__(self, cg_stats=True, spx_stats=True, img_stats=False, verbose=False):
        """
        Constructor for computing dataset statistics

        :param cg_stats: (bool) if compute cell graph stats
        :param spx_stats: (bool) if compute superpx graph stats
        :param img_stats: (bool) if compute image stats
        :param verbose: (bool) if print stats during computation
        """

        self.cg_stats = cg_stats
        self.spx_stats = spx_stats
        self.img_stats = img_stats
        self.verbose = verbose
        self.stats = {}
        self.agg_stats = {}

    def __call__(self, dataloader):
        """
        Compute data statistics

        :param dataloader:
        :return:
        """

        if self.cg_stats:
            self.compute_cell_graph_stats(dataloader)

        if self.spx_stats:
            self.compute_spx_graph_stats(dataloader)

        if self.img_stats:
            self.compute_img_stats(dataloader)

        return self.agg_stats

    def compute_cell_graph_stats(self, dataloader):
        """
        Compute cell graph stats

        :return:
        """

        # define cell graph stats
        self.stats['cell_graph'] = {}

        # compute and store all data stats
        for split, split_data in dataloader.items():
            self.stats['cell_graph'][split] = defaultdict(list)
            for data, label in split_data:
                cell_graph = data[0]
                stats = {
                    'num_nodes': cell_graph.number_of_nodes(),
                    'num_edges': cell_graph.number_of_edges()
                }
                self.stats['cell_graph'][split][label.item()].append(stats)

        # compute summarize stats
        self.agg_stats['cell_graph'] = {}

        for split, stats in self.stats['cell_graph'].items():
            self.agg_stats['cell_graph'][split] = {}
            for cls, stats_per_cls in stats.items():
                self.agg_stats['cell_graph'][split][cls] = {}

                # number of graphs
                num_graphs = len(self.stats['cell_graph'][split][cls])

                # avg & std number of nodes
                avg_num_nodes = sum(map(lambda x: x['num_nodes'], self.stats['cell_graph'][split][cls])) / num_graphs
                std_num_nodes = np.std(list(map(lambda x: x['num_nodes'], self.stats['cell_graph'][split][cls])))

                # avg & std number of edges
                avg_num_edges = sum(map(lambda x: x['num_edges'], self.stats['cell_graph'][split][cls])) / num_graphs
                std_num_edges = np.std(list(map(lambda x: x['num_edges'], self.stats['cell_graph'][split][cls])))

                self.agg_stats['cell_graph'][split][cls]['num_nodes'] = avg_num_nodes
                self.agg_stats['cell_graph'][split][cls]['std_num_nodes'] = std_num_nodes
                self.agg_stats['cell_graph'][split][cls]['num_edges'] = avg_num_edges
                self.agg_stats['cell_graph'][split][cls]['std_num_edges'] = std_num_edges
                self.agg_stats['cell_graph'][split][cls]['num_graphs'] = num_graphs

    def compute_spx_graph_stats(self, dataloader):
        """
        Compute super pixel graph stats

        :return:
        """

        self.stats['superpx_graph'] = {}

    def compute_img_stats(self, dataloader):
        """
        Compute image stats

        :return:
        """

        self.stats['img_graph'] = {}
