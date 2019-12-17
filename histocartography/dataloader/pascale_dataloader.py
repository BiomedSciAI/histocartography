"""Pascale Dataset loader."""
import os
import dgl
import torch.utils.data

from histocartography.dataloader.base_dataloader import BaseDataset
from histocartography.utils.io import get_files_in_folder, complete_path

COLLATE_FN = {
    'DGLGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x
}


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

        if self.model_type == 'CellGraphModel':
            cell_graph = self.cell_graph_builder(objects, image_size)
            return label, cell_graph
        elif self.model_type == 'SuperpxGraphModel':
            superpx_graph = self.superpx_graph_builder(objects, image_size)
            return label, superpx_graph
        elif self.model_type == 'MultiGraphModel':
            cell_graph = self.cell_graph_builder(objects, image_size)
            superpx_graph = self.superpx_graph_builder(objects, image_size)
            assignment_matrix = torch.empty(
                superpx_graph.number_of_nodes(),
                cell_graph.number_of_nodes()
            ).random_(2)
            return cell_graph, superpx_graph, assignment_matrix
        else:
            raise ValueError('Model type: {} not supported'.format(self.model_type))

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
                for filename in get_files_in_folder(path, '.h5')
            ]
        )
    else:
        raise RuntimeError(
            'Provide a folder containing .h5 files.'
        )


def collate(batch):
    """
    Collate a batch.

    Args:
        batch (torch.tensor): a batch of examples.

    Returns:
        a tuple of torch.tensors.
    """

    def collate_fn(batch, id, type):
        return COLLATE_FN[type]([example[id] for example in batch])

    num_modalities = len(batch[0])
    labels = [example[0] for example in batch]
    data = [collate_fn(batch, mod_id, type(batch[0][mod_id]).__name__) for mod_id in range(1, num_modalities)]

    return labels, data


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
