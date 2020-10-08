import numpy as np
import matplotlib.pyplot as plt
import os 


class BoxPlotVisualization:
    """
    Compute the mean/std per nuclei type per tumor type 
    """

    def __init__(self, cuda=False):
        self.cuda = cuda

    def __call__(self, importances_per_tumor_type, name='NucleiViz', save_path=''):
        """
        """
        # for _, val in importances_per_tumor_type.items():
        #     print('Name:', name, len(val['all']))
        fig, ax = plt.subplots()
        ax.set_title(name)
        ax.boxplot([val['all'] for _, val in importances_per_tumor_type.items()])
        plt.savefig(os.path.join(save_path, name))
