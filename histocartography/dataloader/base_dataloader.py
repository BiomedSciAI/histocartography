from torch.utils.data import Dataset
import importlib

from histocartography.graph_building.constants import (
    GRAPH_BUILDING_TYPE, AVAILABLE_GRAPH_BUILDERS,
    GRAPH_BUILDING_MODULE, GRAPH_BUILDING
)


class BaseDataset(Dataset):

    def __init__(self, config, cuda=False):
        """
        Base dataset constructor.
        """
        self._build_graph_builder(config[GRAPH_BUILDING], name='cell_graph_builder')
        self._build_graph_builder(config[GRAPH_BUILDING], name='superpx_graph_builder')
        self.cuda = cuda

    def __getitem__(self, item):
        """

        :param item:
        :return:
        """
        raise NotImplementedError('Implementation in subclasses.')

    def __len__(self):
        """

        :return:
        """
        raise NotImplementedError('Implementation in subclasses.')

    def _build_graph_builder(self, config, name='cell_graph_builder'):
        """
        Build graph builder
        """
        graph_builder_type = config[GRAPH_BUILDING_TYPE]
        if graph_builder_type in list(AVAILABLE_GRAPH_BUILDERS.keys()):
            module = importlib.import_module(
                GRAPH_BUILDING_MODULE.format(graph_builder_type)
            )
            vars(self)[name] = getattr(module, AVAILABLE_GRAPH_BUILDERS[graph_builder_type])(config)
        else:
            raise ValueError(
                'Graph builder type: {} not recognized. Options are: {}'.format(
                    graph_builder_type, list(AVAILABLE_GRAPH_BUILDERS.keys())
                )
            )

