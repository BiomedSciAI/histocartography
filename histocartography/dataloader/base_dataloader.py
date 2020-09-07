from torch.utils.data import Dataset
import importlib

from histocartography.graph_building.constants import (
    GRAPH_BUILDING_TYPE, AVAILABLE_GRAPH_BUILDERS,
    GRAPH_BUILDING_MODULE, GRAPH_BUILDING
)
from histocartography.utils.io import get_device


class BaseDataset(Dataset):

    def __init__(self, config, cuda=False):
        """
        Base dataset constructor.
        """

        self.cuda = cuda
        self.device = get_device(self.cuda)
        self.model_type = config['model_type']

        # for graph_type, param in config[GRAPH_BUILDING].items():
        #     self._build_graph_builder(param, name=graph_type)

    def __getitem__(self, item):
        """
        Get an example.
        """
        raise NotImplementedError('Implementation in subclasses.')

    def __len__(self):
        """
        Num samples in the dataset.
        """
        raise NotImplementedError('Implementation in subclasses.')

    def _build_graph_builder(self, config, name):
        """
        Build graph builder
        """
        graph_builder_type = config[GRAPH_BUILDING_TYPE]
        if graph_builder_type in list(AVAILABLE_GRAPH_BUILDERS.keys()):
            module = importlib.import_module(
                GRAPH_BUILDING_MODULE.format(graph_builder_type)
            )
            vars(self)[name] = getattr(
                module, AVAILABLE_GRAPH_BUILDERS[graph_builder_type])(config)
        else:
            raise ValueError(
                'Graph builder type: {} not recognized. Options are: {}'.format(
                    graph_builder_type, list(
                        AVAILABLE_GRAPH_BUILDERS.keys()
                    )
                )
            )
