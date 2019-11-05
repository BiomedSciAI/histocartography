
class BaseGraphBuilder:
    """
    Base interface class for graph building.
    """

    def __init__(self, config, cuda=False, verbose=False):
        """
        Base Graph Builder constructor.

        Args:
            config: list of required params to build a graph
        """

        super(BaseGraphBuilder, self).__init__()

        if verbose:
            print('*** Build base graph builder ***')

        self.config = config
        self.cuda = cuda

    def build_graph(self, data):
        # @TODO define the input argument
        """
        Build graph
        :param data:
        """
