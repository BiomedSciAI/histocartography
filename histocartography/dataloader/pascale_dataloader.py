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
            data_path,
            dataset_name,
            split,
            config,
            cuda=False,
            load_cell_graph=True,
            load_superpx_graph=True,
            load_image=False,
    ):
        """
        Pascale dataset constructor.

        Args:
            :param config: (dict) config file
            :param dir_path (str): path to the pascale dataset and split files
            :param cuda (bool): cuda usage
            :param norm_cell_features (bool): if the cell features should be normalised
            :param norm_superpx_features (bool): if the super pixel features should be normalised
            :param split(str): train, validation or test
        """
        super(PascaleDataset, self).__init__(config, cuda)

        print('Start loading dataset {} : {}'.format(dataset_name, split))

        # 1. set class attributes
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.load_cell_graph = load_cell_graph
        self.load_superpx_graph = load_superpx_graph
        self.load_image = load_image

        # 2. load h5 fnames and labels (from h5 fname)
        self._load_h5_fnames_and_labels(data_path, split)

        # 3. extract meta data
        self.num_samples = len(self.h5_fnames)
        if load_cell_graph:
            self.num_cell_features = self._get_cell_features_dim()
            self._build_normaliser(graph_type='cell_graph')
        if load_superpx_graph:
            self.num_superpx_features = 10
            self._build_normaliser(graph_type='superpx_graph')

        # 4. load the images if path
        if load_image:
            self._load_image_fnames()

    def _load_image_fnames(self, img_path):
        self.image_fnames = get_files_in_folder(complete_path(img_path, self.dataset_name), 'png')

    def _load_h5_fnames_and_labels(self, data_path, train_flag):
        """
        Load the h5 data from the text files
        """
        self.labels = []
        extension = '.h5'

        self.h5_fnames = get_files_from_text(path, extension, train_flag)

        for fname in self.h5_fnames:
            self._load_label(complete_path(data_path, fname))

    def _load_label(self, fpath):
        """
        Load the label by inspecting the filename
        """
        tumor_type = list(
            filter(
                lambda x: x in fpath, list(
                    TUMOR_TYPE_TO_LABEL.keys())))[0]
        self.labels.append(TUMOR_TYPE_TO_LABEL[tumor_type])

    def _get_cell_features_dim(self):
        with h5py.File(self.h5_fnames[0], 'r') as f:
            cell_features = h5_to_tensor(f['instance_features'], self.device)
            centroid = h5_to_tensor(f['instance_centroid_location'], self.device)
            f.close()
        return cell_features.shape[1] + centroid.shape[1]

    def _build_normaliser(self, graph_type):
        """
        Build normalizers to normalize the node features (ie, mean=0, std=1)
        """
        if not NORMALIZATION_FACTORS[graph_type]:
            self.cell_features_transformer = compute_normalization_factor(
                self.data_path, self.h5_fnames)
        else:
            self.cell_features_transformer = NORMALIZATION_FACTORS[graph_type]

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

    def _build_cell_graph(self, index):
        """
        Build the cell graph
        """
        # extract the image size, centroid, cell features and label
        with h5py.File(self.h5_fnames[index], 'r') as f:
            image_size = h5_to_tensor(f['image_dimension'], self.device)
            cell_features = h5_to_tensor(f['instance_features'], self.device)
            centroid = h5_to_tensor(f['instance_centroid_location'], self.device)
            image_size = image_size.type(torch.float32)
            norm_centroid = centroid / image_size[:-1]
            f.close()

        # normalize the cell features
        cell_features = \
            (cell_features - self.cell_features_transformer['mean'].to(self.device)) / \
            (self.cell_features_transformer['std']).to(self.device)

        # concat spatial + appearance features
        cell_features = torch.cat((cell_features, norm_centroid), dim=1)

        # build topology
        cell_graph = self.cell_graph_builder(cell_features, centroid)

        return cell_graph

    def __getitem__(self, index):
        """
        Get an example.

        Args:
            index (int): index of the example.
        Returns:
        """

        # 1. load label
        label = self.labels[index]

        data = []

        # 2. load cell graph
        if self.load_cell_graph:
            cell_graph = self._build_cell_graph(index)
            data.append(cell_graph)

        # 3. load superpx graph
        if self.load_superpx_graph:
            superpx_graph = self._build_superpx_graph(index)
            data.append(superpx_graph)

        # 4. load assignment matrix to go from the cell graph to the the superpx graph
        if self.load_cell_graph and self.load_superpx_graph:
            assignment_matrix = self._build_assignment_matrix(cell_graph, superpx_graph)
            data.append(assignment_matrix)

        # 4. load the image if required
        if self.load_image:
            image, image_name = self._load_image(self.h5_fnames[index].replace('.h5', ''))
            data.append(image_name)
            data.append(image_name)

        return data, label

    def __len__(self):
        """Return the number of samples in the WSI."""
        return self.num_samples


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
        return COLLATE_FN[type]([example[0][id] for example in batch])

    # collate the data
    num_modalities = len(batch[0][0])
    data = tuple([collate_fn(batch, mod_id, type(batch[0][0][mod_id]).__name__)
                  for mod_id in range(num_modalities)])

    # collate the labels
    labels = torch.LongTensor([example[1] for example in batch])

    return data, labels


def build_datasets(path, *args, **kwargs):
    """
    Builds dataset from text files that contain train:test:validation split

    Returns:
        PASCALE datasets for train, validation and testing
    """

    # TODO : check if this is fine: to add directory to path
    path = complete_path(path, 'nuclei_info')
    data_dir = [f.path for f in os.scandir(path) if f.is_dir()]
    data_dir = list(filter(lambda x: all(b not in x for b in DATASET_BLACKLIST), data_dir))

    datasets = {}

    for data_split in ['train', 'val', 'test']:
        datasets[data_split] = torch.utils.data.ConcatDataset(
            datasets=[
                PascaleDataset(
                    complete_path(dir, '_h5'),
                    split(dir)[-1],
                    data_split,
                    *args, **kwargs
                )
                for dir in data_dir
            ]
        )

    return datasets


def make_data_loader(
        batch_size,
        num_workers=0,
        *args,
        **kwargs):
    """
    Create a data loader for the dataset.

    Args:
        batch_size (int): size of the batch.
        num_workers (int): number of workers.
    Returns:
        dataloaders: a dict containing the train and val data loader.
        num_cell_features: num of cell features in the cell graph. Data dependent parameters
            required by the model
    """

    datasets = build_datasets(*args, **kwargs)
    dataloaders = {}

    for split, data in datasets.items():
        dataloaders[split] = torch.utils.data.DataLoader(
                data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate
            )
        num_features = data.datasets[0].num_cell_features

    return dataloaders, num_features
