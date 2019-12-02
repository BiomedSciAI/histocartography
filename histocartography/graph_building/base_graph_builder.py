# @TODO: call this function in the __get_item__() of the dataloader.


class BaseGraphBuilder:
    """
    Base interface class for graph building.
    """

    def __init__(self, config, cuda=False, verbose=False):
        """
        Base Graph Builder constructor.

        Args:
            config: (dict) graph building params
            cuda: (bool) if cuda is available
            verbose: (bool) verbosity level
        """
        super(BaseGraphBuilder, self).__init__()

        if verbose:
            print('*** Build base graph builder ***')

        self.config = config
        self.cuda = cuda
        self.verbose = verbose

    def __call__(self, objects):
        """
        Build graph
        Args:
            objects: (list) each element in the list is a dict with:
                - bbox
                - label
                - visual descriptor
        """

    def _set_node_features(self, objects):
        """
        Build node embeddings
        :return:
        """

    def _set_edge_embeddings(self, objects):
        """
        Build edge embedding
        :return:
        """

    def _build_topology(self, objects):
        """
        Build topology
        :return:
        """