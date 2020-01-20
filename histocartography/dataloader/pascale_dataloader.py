"""Pascale Dataset loader."""
import os
from os.path import split
import torch.utils.data
import h5py
from histocartography.dataloader.base_dataloader import BaseDataset
from histocartography.utils.io import get_files_in_folder, complete_path, h5_to_tensor, get_files_from_text
from histocartography.dataloader.constants import (
    NORMALIZATION_FACTORS, COLLATE_FN,
    TUMOR_TYPE_TO_LABEL, DATASET_BLACKLIST
)
from histocartography.utils.vector import compute_normalization_factor
from histocartography.utils.io import load_image


class PascaleDataset(BaseDataset):
    """Pascale data loader."""

    def __init__(
            self,
            dir_path,
            dataset_name,
            status,
            text_path,
            config,
            cuda=False,
            norm_cell_features=True,
            norm_superpx_features=False,
            img_path=None,
    ):
        """
        Pascale dataset constructor.

        Args:
            :param config: (dict) config file
            :param dir_path (str): path to the pascale dataset
            :param cuda (bool): cuda usage
            :param norm_cell_features (bool): if the cell features should be normalised
            :param norm_superpx_features (bool): if the super pixel features should be normalised
            :param status(str): train, validation or test
            :param text_path(str): path to text files containing train:validation:test split
        """
        super(PascaleDataset, self).__init__(config, cuda)

        print('Start loading dataset {} : {}'.format(dataset_name, status))

        # 1. load and store h5 data
        self.dir_path = dir_path
        self.dataset_name = dataset_name
        self.img_path = img_path
        self.text_path = text_path
        self.status = status

        if text_path is not None:
            self._load_files(text_path, dir_path, status)
        else:
            self._load_and_store_dataset(dir_path)

        # 2. extract meta info from data
        self.num_samples = len(self.h5_fnames)
        self.num_cell_features = self._get_cell_features_dim()

        # 3. build data normalizer
        self.norm_cell_features = norm_cell_features
        self.norm_superpx_features = norm_superpx_features
        self._build_normaliser()

        # 4. load the images if path
        if img_path is not None:
            self._load_images(img_path)

    def _load_images(self, img_path):
        self.image_fnames = get_files_in_folder(complete_path(img_path, self.dataset_name), 'png')

    def _get_cell_features_dim(self):

        with h5py.File(complete_path(self.dir_path, self.h5_fnames[0]), 'r') as f:
            cell_features = h5_to_tensor(f['instance_features'], self.device)
            centroid = h5_to_tensor(f['instance_centroid_location'], self.device)
            f.close()
        return cell_features.shape[1] + centroid.shape[1]

    def _build_normaliser(self):
        """
        Build normalizers to normalize the node features (ie, mean=0, std=1)
        """
        if self.norm_cell_features:
            if not NORMALIZATION_FACTORS['cell_graph']:
                self.cell_features_transformer = compute_normalization_factor(
                    self.dir_path, self.h5_fnames)
            else:
                self.cell_features_transformer = NORMALIZATION_FACTORS['cell_graph']

        if self.norm_superpx_features:
            raise NotImplementedError(
                "Super pixel normalization not implemented.")

    def _load_and_store_dataset(self, dir_path):
        """
        Load the h5 data and store them in lists
        """
        self.labels = []
        self.h5_fnames = get_files_in_folder(dir_path, 'h5')

        for fname in self.h5_fnames:
            self._load_label(complete_path(dir_path, fname))

    def _load_files(self, text_path, path, train_flag):
        """
        Load the h5 data from the text files
        """
        self.labels = []
        extension = '.h5'

        self.h5_fnames = get_files_from_text(path, text_path, extension, train_flag)

        for fname in self.h5_fnames:
            self._load_label(complete_path(path, fname))

    def _load_label(self, fpath):
        """
        Load the label by inspecting the filename
        """
        tumor_type = list(
            filter(
                lambda x: x in fpath, list(
                    TUMOR_TYPE_TO_LABEL.keys())))[0]
        self.labels.append(TUMOR_TYPE_TO_LABEL[tumor_type])

    def _load_sample(self, fpath):
        """
        Load a h5 dataset
        """
        with h5py.File(fpath, 'r') as f:
            # extract features, centroid and image dimension
            node_embeddings = h5_to_tensor(f['instance_features'], self.device)
            centroid = h5_to_tensor(
                f['instance_centroid_location'], self.device)
            image_dim = h5_to_tensor(f['image_dimension'], self.device)

            # append
            self.cell_features.append(node_embeddings)
            self.cell_centroids.append(centroid)
            self.image_dimensions.append(image_dim)
            f.close()

    def _load_image(self, img_name):
        if img_name + '.png' in self.image_fnames:
            return load_image(
                complete_path(
                    complete_path(
                        self.img_path, self.dataset_name
                    ), img_name + '.png'
                )
            ), img_name
        else:
            print('Warning: the image {} doesnt seem to exist in path {}'.format(img_name, self.img_path))

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

        # extract the image size, centroid, cell features and label
        with h5py.File(complete_path(self.dir_path, self.h5_fnames[index]), 'r') as f:
            image_size = h5_to_tensor(f['image_dimension'], self.device)
            cell_features = h5_to_tensor(f['instance_features'], self.device)
            norm_centroid = h5_to_tensor(f['instance_centroid_location'], self.device) / image_size[:-1]
            centroid = h5_to_tensor(f['instance_centroid_location'], self.device)

            f.close()

        label = self.labels[index]

        # load the image if required
        if self.img_path is not None:
            image, image_name = self._load_image(self.h5_fnames[index].replace('.h5', ''))

        # normalize the appearance-based cell features
        if self.norm_cell_features:
            cell_features = \
                (cell_features - self.cell_features_transformer['mean']) / \
                self.cell_features_transformer['std']

        # concat spatial + appearance features
        cell_features = torch.cat((cell_features, norm_centroid), dim=1)

        # build graph topology
        if self.model_type == 'cell_graph_model':
            cell_graph = self.cell_graph_builder(cell_features, centroid)
            if self.img_path is not None:
                return cell_graph, image, image_name, label
            return cell_graph, label
        else:
            raise ValueError(
                'Model type: {} not supported'.format(
                    self.model_type))

    def __len__(self):
        """Return the number of samples in the WSI."""
        return self.num_samples


def build_dataset(path, *args, **kwargs):
    """
    Build the dataset.

    Returns:
        a PascaleDataset.
    """

    # 1. list all dir in path and remove the dataset from our blacklist
    data_dir = [f.path for f in os.scandir(path) if f.is_dir()]
    data_dir = list(filter(lambda x: all(b not in x for b in DATASET_BLACKLIST), data_dir))

    # 2. build dataset by concatenating all the sub-datasets
    if os.path.isdir(path):
        return torch.utils.data.ConcatDataset(
            datasets=[
                PascaleDataset(
                    complete_path(dir, '_h5'),
                    split(dir)[-1], "all", None,
                    *args, **kwargs
                )
                for dir in data_dir
            ]
        )
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
    data = tuple([collate_fn(batch, mod_id, type(batch[0][mod_id]).__name__)
                  for mod_id in range(0, num_modalities - 1)])
    labels = torch.LongTensor([example[-1] for example in batch])

    return data, labels


def build_dataset_from_text(text_path, path, *args, **kwargs):
    """
    Builds dataset from text files that contain train:test:validation split

    Returns:
        Two PASCALE datasets for train and validation
    """

    data_dir = [f.path for f in os.scandir(path) if f.is_dir()]
    data_dir = list(filter(lambda x: all(b not in x for b in DATASET_BLACKLIST), data_dir))

    # 2. build dataset by concatenating all the sub-datasets
    if os.path.isdir(path):
        train_data = torch.utils.data.ConcatDataset(
            datasets=[
                PascaleDataset(
                    complete_path(dir, '_h5'),
                    split(dir)[-1], "train",
                    text_path,
                    *args, **kwargs
                )
                for dir in data_dir
            ]
        )
        valid_data = torch.utils.data.ConcatDataset(
            datasets=[
                PascaleDataset(
                    complete_path(dir, '_h5'),
                    split(dir)[-1], "valid",
                    text_path,
                    *args, **kwargs
                )
                for dir in data_dir
            ]
        )
        test_data = torch.utils.data.ConcatDataset(
            datasets=[
                PascaleDataset(
                    complete_path(dir, '_h5'),
                    split(dir)[-1], "test",
                    text_path,
                    *args, **kwargs
                )
                for dir in data_dir
            ]
        )
    else:
        raise RuntimeError(
            '{} doesnt seem to exist.'.format(path)
        )
    return train_data, valid_data, test_data


def _build_dataset_loaders(train_data, validation_data, batch_size, workers):
    """

    Returns the dataset loader for train and validation
    """
    dataset_loader = {
        'train':
            torch.utils.data.DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=workers,
                collate_fn=collate
            ),
        'val':
            torch.utils.data.DataLoader(
                validation_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                collate_fn=collate
            )
    }
    return dataset_loader


def make_data_loader(
        batch_size,
        text_path,
        train_ratio,
        num_workers=1,
        *args,
        **kwargs):
    """
    Create a data loader for the dataset.

    Args:
        batch_size (int): size of the batch.
        text_path(str) : path to text files containing split
        train_ratio(float) : if text_path not given, randomly splits dataset as per ratio
        num_workers (int): number of workers.
    Returns:
        dataloaders: a dict containing the train and val data loader.
        num_cell_features: num of cell features in the cell graph. Data dependent parameters
            required by the model
    """

    if text_path:
        training_dataset, val_dataset, test_dataset = build_dataset_from_text(text_path, *args, **kwargs)
        dataset_loaders = _build_dataset_loaders(training_dataset, val_dataset, batch_size, num_workers)
        dataset_loaders.update({
            'test':
                torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    collate_fn=collate
                    )
            })
        num_features = training_dataset.datasets[0].num_cell_features
    else:
        full_dataset = build_dataset(*args, **kwargs)
        train_length = int(len(full_dataset) * train_ratio)
        val_length = len(full_dataset) - train_length
        training_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_length, val_length])
        dataset_loaders = _build_dataset_loaders(training_dataset, val_dataset, batch_size, num_workers)
        num_features = full_dataset.datasets[0].num_cell_features

    return dataset_loaders, num_features
