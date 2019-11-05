from histocartography.graph_building.base_graph_builder import BaseGraphBuilder


class KNNGraphBuilder(BaseGraphBuilder):
    """
    KNN (K-Nearest Neighbors) class for graph building.
    """

    def __init__(self, config, cuda=False, verbose=False):
        """
        Base Graph Builder constructor.

        Args:
            config: list of required params to build a graph
        """

        super(KNNGraphBuilder, self).__init__(config)

        if verbose:
            print('*** Build k-NN graph ***')

        self.config = config
        self.cuda = cuda

    def build_graph(self, data):
        """
        Build graph
        :param data:
        """
