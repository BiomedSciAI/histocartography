"""Pascale Dataset loader."""
import os
import dgl
import torch.utils.data
import importlib

from histocartography.dataloader.base_dataloader import BaseDataset
from histocartography.utils.io import get_files_in_folder, complete_path
from histocartography.graph_building.constants import (
    GRAPH_BUILDING_TYPE, AVAILABLE_GRAPH_BUILDERS,
    GRAPH_BUILDING_MODULE, GRAPH_BUILDING
)


class PascaleDataset(BaseDataset):
    """Pascale data loader."""

    def __init__(self, fpath, config, cuda=False):
        """
        Pascale dataset constructor.

        Args:
            config: (dict) config file
            fpath (str): path to a pascale dataset whole slide image.
            cuda (bool): cuda usage.
        """
        super(PascaleDataset, self).__init__(cuda)
        self._load_dataset(fpath)
        self._build_graph_builder(config[GRAPH_BUILDING])

    def _build_graph_builder(self, config):
        """
        Build graph builder
        """
        graph_builder_type = config[GRAPH_BUILDING_TYPE]
        if graph_builder_type in list(AVAILABLE_GRAPH_BUILDERS.keys()):
            module = importlib.import_module(
                GRAPH_BUILDING_MODULE.format(graph_builder_type)
            )
            self.graph_builder = getattr(module, AVAILABLE_GRAPH_BUILDERS[graph_builder_type])(config)
        else:
            raise ValueError(
                'Graph builder type: {} not recognized. Options are: {}'.format(
                    graph_builder_type, list(AVAILABLE_GRAPH_BUILDERS.keys())
                )
            )

    def _load_dataset(self, fpath):
        """
        Load WSI tumor regions
        """
        self.samples = ...

    def __getitem__(self, index):
        """
        Get an example.

        Args:
            index (int): index of the example.
        Returns:
            a tuple containing: @TODO can be changed
                 - dgl graph,
                 - image
                 - labels.
        """

        objects = ...
        image_size = ...
        label = ...
        image = ...

        g = self.graph_builder(objects, image_size)

        return g, image, label

    def __len__(self):
        """Return the number of samples in the WSI."""
        return len(self.samples)


def build_dataset(path, *args, **kwargs):
    """
    Build the dataset.

    Returns:
        a PascaleDataset.
    """
    if os.path.isdir(path):
        return torch.utils.data.ConcatDataset(
            datasets=[
                PascaleDataset(
                    complete_path(path, filename),
                    *args, **kwargs
                )
                for filename in get_files_in_folder(path, '.svs')
            ]
        )
    else:
        raise RuntimeError(
            'Provide a folder containing .svs files.'
        )


def collate(batch):
    """
    Collate a batch.

    Args:
        batch (torch.tensor): a batch of examples.

    Returns:
        a tuple of torch.tensors.
    """
    graphs = dgl.batch([example[0] for example in batch])
    images = [example[1] for example in batch]
    labels = [example[2] for example in batch]
    return graphs, images, labels


def make_data_loader(batch_size, num_workers=1, *args, **kwargs):
    """
    Create a data loader for the dataset.

    Args:
        batch_size (int): size of the batch.
        num_workers (int): number of workers.
    Returns:
        a tuple containing the data loader and the dataset.
    """
    dataset = build_dataset(*args, **kwargs)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers
    )
    return data_loader, dataset
