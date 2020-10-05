import numpy as np
import matplotlib.pyplot as plt


class BoxPlotVisualization:
    """
    Compute the mean/std per nuclei type per tumor type 
    """

    def __init__(self, cuda=False):
        super(BoxPlotVisualization, self).__init__(cuda)

    def __call__(self, importances_per_tumor_type):  # node importance ?
        """
        """
        fig, ax = plt.subplots()
        ax.set_title('Multiple Samples with Different sizes')
        ax.boxplot([val for _, val in importances_per_tumor_type.items()])
        plt.show()
