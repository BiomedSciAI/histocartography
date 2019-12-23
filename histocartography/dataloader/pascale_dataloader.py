"""Pascale Dataset loader."""
import os
import torch.utils.data
import h5py
from histocartography.dataloader.base_dataloader import BaseDataset
from histocartography.utils.io import get_files_in_folder, complete_path, h5_to_tensor
from histocartography.dataloader.constants import NORMALIZATION_FACTORS, COLLATE_FN, TUMOR_TYPE_TO_LABEL
from histocartography.utils.vector import compute_normalization_factor


class PascaleDataset(BaseDataset):
    """Pascale data loader."""

    def __init__(self, dir_path, config, cuda=False, norm_cell_features=True, norm_superpx_features=False):
        """
        Pascale dataset constructor.

        Args:
            :param config: (dict) config file
            :param dir_path (str): path to the pascale dataset.
            :param cuda (bool): cuda usage.
            :param norm_cell_features (bool): if the cell features should be normalised
            :param norm_superpx_features (bool): if the super pixel features should be normalised
        """
        super(PascaleDataset, self).__init__(config, cuda)

        # 1. load and store h5 data
        self._load_and_store_dataset(dir_path)

        # 2. extract meta info from data
        self.num_samples = len(self.image_dimensions)
        self.num_cell_features = self.cell_features[0].shape[1] + self.cell_centroids[0].shape[1]

        # 3. build data normalizer
        self.norm_cell_features = norm_cell_features
        self.norm_superpx_features = norm_superpx_features
        self._build_normaliser()

    def _build_normaliser(self):
        """
        Build normalizer to normalize the node features (ie, mean=0, std=1)
        """
        if self.norm_cell_features:
            if not NORMALIZATION_FACTORS['cell_graph']:
                self.cell_features_transformer = compute_normalization_factor(self.cell_features)
            else:
                self.cell_features_transformer = NORMALIZATION_FACTORS['cell_graph']

        if self.norm_superpx_features:
            raise NotImplementedError("Super pixel normalization not implemented.")

    def _load_and_store_dataset(self, dir_path):
        """
        Load the h5 data and store them in lists
        """
        self.cell_features = []
        self.cell_centroids = []
        self.image_dimensions = []
        self.labels = []

        h5_fnames = get_files_in_folder(dir_path, 'h5')

        for fname in h5_fnames:
            self._load_sample(complete_path(dir_path, fname))
            self._load_label(fname)

    def _load_label(self, fpath):
        """
        Load the label by inspecting the filename
        """
        tumor_type = list(filter(lambda x: x in fpath, list(TUMOR_TYPE_TO_LABEL.keys())))[0]
        self.labels.append(TUMOR_TYPE_TO_LABEL[tumor_type])

    def _load_sample(self, fpath):
        """
        Load a h5 dataset
        """
        with h5py.File(fpath, 'r') as f:
            # extract features, centroid and image dimension
            node_embeddings = h5_to_tensor(f['instance_features'], self.device)
            centroid = h5_to_tensor(f['instance_centroid_location'], self.device)
            image_dim = h5_to_tensor(f['image_dimension'], self.device)

            # append
            self.cell_features.append(node_embeddings)
            self.cell_centroids.append(centroid)
            self.image_dimensions.append(image_dim)
            f.close()

    def __getitem__(self, index):
        """
        Get an example.

        Args:
            index (int): index of the example.
        Returns:
            a tuple containing:
                 - cell_graph (dgl graph)
                 - @TODO: superpx_graph (dgl graph)
                 - @TODO: assignment matrix (list of LongTensor)
                 - labels (LongTensor)
        """

        image_size = self.image_dimensions[index]
        centroid = self.cell_centroids[index] / image_size[:-1]
        cell_features = self.cell_features[index]
        label = self.labels[index]

        # normalize the appearance-based cell features
        if self.norm_cell_features:
            cell_features = \
                (cell_features - self.cell_features_transformer['mean']) / \
                self.cell_features_transformer['std']

        # concat spatial + appearance features
        cell_features = torch.cat((cell_features, centroid), dim=1)

        if self.model_type == 'cell_graph_model':
            cell_graph = self.cell_graph_builder(cell_features, centroid)
            return cell_graph, label
        else:
            raise ValueError('Model type: {} not supported'.format(self.model_type))

    def __len__(self):
        """Return the number of samples in the WSI."""
        return self.num_samples


def build_dataset(path, *args, **kwargs):
    """
    Build the dataset.

    Returns:
        a PascaleDataset.
    """
    if os.path.isdir(path):
        return PascaleDataset(path, *args, **kwargs)
    else:
        raise RuntimeError(
            '{} doesnt seem to exist.'.format(path)
        )


def collate(batch):
    """
    Collate a batch.

    Args:
        batch (torch.tensor): a batch of examples.

    Returns:
        data: (tuple)
        labels: (torch.LongTensor)
    """

    def collate_fn(batch, id, type):
        return COLLATE_FN[type]([example[id] for example in batch])

    num_modalities = len(batch[0])
    data = tuple([collate_fn(batch, mod_id, type(batch[0][mod_id]).__name__) for mod_id in range(0, num_modalities-1)])
    labels = torch.LongTensor([example[-1] for example in batch])

    return data, labels


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
                num_workers=num_workers,
                collate_fn=collate
            ),
        'val':
            torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate
            )
    }

    return dataset_loaders, full_dataset.num_cell_features
