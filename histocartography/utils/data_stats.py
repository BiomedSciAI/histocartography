from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from histocartography.dataloader.constants import LABEL_TO_TUMOR_TYPE
from histocartography.utils.io import complete_path


LABEL_TO_COLOR = {
    '0': 'green',
    '1': 'cyan',
    '2': 'red',
    '3': 'yellow',
    '4': 'orange'
}


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

        return self.agg_stats, self.stats

    def compute_cell_graph_stats(self, dataloader):
        """
        Compute cell graph stats

        :return:
        """

        # 1. define cell graph stats
        self.stats['cell_graph'] = {}

        # compute and store all data stats
        for split, split_data in dataloader.items():
            self.stats['cell_graph'][split] = defaultdict(list)
            for data, label in tqdm(split_data):
                cell_graph = data[0]
                stats = {
                    'num_nodes': cell_graph.number_of_nodes(),
                    'num_edges': cell_graph.number_of_edges()
                }
                self.stats['cell_graph'][split][label.item()].append(stats)

        # 2. compute per class agg stats
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

        # 3. compute agg stats
        for split, stats in self.stats['cell_graph'].items():

            num_graphs = sum(map(lambda x: x['num_graphs'], self.agg_stats['cell_graph'][split].values()))
            avg_num_nodes = sum(map(lambda x: x['num_graphs'] * x['num_nodes'], self.agg_stats['cell_graph'][split].values())) / num_graphs
            avg_num_edges = sum(map(lambda x: x['num_graphs'] * x['num_edges'], self.agg_stats['cell_graph'][split].values())) / num_graphs

            self.agg_stats['cell_graph'][split]['all'] = {}
            self.agg_stats['cell_graph'][split]['all']['num_nodes'] = avg_num_nodes
            self.agg_stats['cell_graph'][split]['all']['num_edges'] = avg_num_edges
            self.agg_stats['cell_graph'][split]['all']['num_graphs'] = num_graphs

    @staticmethod
    def plot_histogram(data, feature, out_dir='', show=True, xlim=10000):
        """

        :param data: (dict) --> should contain stats from the N classes
        :param feature: (str) feature to plot, e.g., num_nodes, num_edges
        :param out_dir: (str) where to save the histograms
        :param show: (bool) if we show the histogram
        :return:
        """

        # 1- make sure that there is no all key
        if 'all' in list(data.keys()):
            del data['all']

        # 2- draw the histogram for each class
        for cls, data_per_class in data.items():
            out = list(map(lambda x: x[feature], data_per_class))
            plt.hist(out, 50, facecolor=LABEL_TO_COLOR[cls], alpha=1., label=LABEL_TO_TUMOR_TYPE[cls])

            plt.xlabel(feature)
            plt.ylabel('Occurence')
            plt.title('{} distribution for each class'.format(feature))
            plt.legend()
            plt.xlim((0, xlim))
            plt.grid(True)

            # 3- save the histogram
            if out_dir:
                plt.savefig(complete_path(out_dir, LABEL_TO_TUMOR_TYPE[cls] + '.png'))

            # 4- show the histogram
            if show:
                plt.show()

            plt.close()

    @staticmethod
    def corr_matrix(data, feature):

        all_df = []

        for cls, data_per_class in data.items():
            out = list(map(lambda x: x[feature], data_per_class))

            data = {
                    feature: out,
                    'label': [int(cls)] * len(out),
                    }

            all_df.append(pd.DataFrame(data))

        df = pd.concat(all_df)
        df = pd.get_dummies(df, columns=['label'], prefix=['1hot_'])
        corr_matrix = df.corr()

        print('Correlation matrix:', corr_matrix)

        return corr_matrix

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
