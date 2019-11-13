"""Consep Dataset loader."""
import dgl
import torch.utils.data

from histocartography.dataloader.base_dataset import BaseDataset
from histocartography.utils.io import get_files_in_folder, load_json, complete_path, load_image


class ConsepDataset(BaseDataset):
    """Consep data loader."""

    def __init__(
        self, filepath, cuda=False, is_train=False
    ):
        """
        Initialize ConsepDataset.

        Args:
            filepath (str): path to the Consep dataset.
            graph_name (str): name of the graph.
            cuda (bool): cuda usage.
            is_train (bool): training dataset.
        """
        super(ConsepDataset, self).__init__(cuda)
        self.is_train = is_train
        self._load_dataset(filepath)

    def _load_dataset(self, path):
        # 1. load annotations
        ann_fnames = get_files_in_folder(path, 'json')
        self.segmentation_annotations = [load_json(complete_path(path, fname)) for fname in ann_fnames]
        # 2. load images
        image_fnames = get_files_in_folder(path, 'png')
        self.images = [load_image(complete_path(path, fname)) for fname in image_fnames]

    def __getitem__(self, index):
        """
        Get an example.

        Args:
            index (int): index of the example.
        Returns:
            a tuple containing:
                 - dgl graph,
                 - image
                 - labels.
        """
        g = dgl.DGLGraph()
        image = self.images[index]
        label = 0
        return g, image, label

    def __len__(self):
        """Return the number of examples."""
        return self.num_datasets


def build_dataset(path, *args, **kwargs):
    """
    Build the dataset.

    Returns:
        an H5Dataset.
    """
    return ConsepDataset(path, *args, **kwargs)


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
