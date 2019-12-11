"""Pascale Dataset loader."""
import os
import dgl
import torch.utils.data

from histocartography.dataloader.base_dataloader import BaseDataset
from histocartography.utils.io import get_files_in_folder, complete_path


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
        super(PascaleDataset, self).__init__(config, cuda)
        self._load_dataset(fpath)

    def _load_dataset(self, fpath):
        """
        Load WSI tumor regions
        """
        self.samples = []

    def __getitem__(self, index):
        """
        Get an example.

        Args:
            index (int): index of the example.
        Returns:
            a tuple containing:
                 - cell_graph (dgl graph)
                 - superpx_graph (dgl graph)
                 - assignment matrix (list of LongTensor)
                 - labels (LongTensor)
        """

        objects = []
        image_size = [1000, 1000]
        label = torch.LongTensor([0])

        cell_graph = self.cell_graph_builder(objects, image_size)
        superpx_graph = self.superpx_graph_builder
        assignment_matrix = torch.empty(
            superpx_graph.number_of_nodes(),
            cell_graph.number_of_nodes()
        ).random_(2)

        return cell_graph, superpx_graph, assignment_matrix, label

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
    cell_graphs = dgl.batch([example[0] for example in batch])
    superpx_graphs = dgl.batch([example[1] for example in batch])
    assignment_matrix = [example[2] for example in batch]
    labels = [example[3] for example in batch]
    return cell_graphs, superpx_graphs, assignment_matrix, labels


def make_data_loader(batch_size, train_ratio=0.8, num_workers=1, *args, **kwargs):
    """
    Create a data loader for the dataset.

    Args:
        batch_size (int): size of the batch.
        num_workers (int): number of workers.
    Returns:
        a tuple containing the data loader and the dataset.
    """

    full_dataset = build_dataset(*args, **kwargs)
    train_length = int(len(full_dataset) * train_ratio)
    val_length = len(full_dataset) - train_length
    training_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_length, val_length]
    )

    dataset_loaders = {
        'train':
            torch.utils.data.DataLoader(
                training_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            ),
        'val':
            torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
    }

    return dataset_loaders
