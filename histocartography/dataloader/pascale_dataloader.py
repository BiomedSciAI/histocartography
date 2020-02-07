"""Pascale Dataset loader."""
import os
import h5py
import torch.utils.data
import numpy as np

from histocartography.dataloader.base_dataloader import BaseDataset
from histocartography.utils.io import (
    complete_path, h5_to_tensor,
    load_h5_fnames, get_dir_in_folder
)
from histocartography.dataloader.constants import (
    NORMALIZATION_FACTORS, COLLATE_FN,
    TUMOR_TYPE_TO_LABEL, DATASET_BLACKLIST)
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
            self.cell_graph_path = os.path.join(
                self.data_path, 'nuclei_info', self.dataset_name, '_h5'
            )
            self.num_cell_features = self._get_cell_features_dim()
            self._build_normaliser(graph_type='cell_graph')

        if load_superpx_graph:
            self.superpx_graph_path = os.path.join(
                self.data_path, 'super_pixel_info','main_sp', 'prob_thr_0.8', self.dataset_name)
            # Path to dgl graphs
            self.graph_path = os.path.join(self.data_path, 'super_pixel_info', 'dgl_graphs', 'prob_thr_0.8',
                                           self.dataset_name)
            # get indices of features
            self.top_feat_path = os.path.join(self.data_path, 'misc_utils', 'main_sp_classification',
                                              'sp_classifier')
            self.top_feat_ind = self._load_feat_indices(self.top_feat_path)

            self.num_superpx_features = self._get_superpx_features_dim()
            self._build_normaliser(graph_type='superpx_graph')

        if load_image:
            self.image_path = os.path.join(
                self.data_path, 'Images_norm', self.dataset_name
            )

    def _load_h5_fnames_and_labels(self, data_path, split):
        """
        Load the h5 data from the text files
        """
        self.labels = []
        extension = '.h5'
        tumor_type = self.dataset_name
        self.h5_fnames = load_h5_fnames(data_path, tumor_type, extension, split)

        for fname in self.h5_fnames:
            self._load_label(fname)

    def _load_graph(self, graph_name):
        """
        Load dgl graphs from path
        """
        return os.path.join(self.graph_path, graph_name + '.bin')

    def _load_label(self, fpath):
        """
        Load the label by inspecting the filename
        """
        tumor_type = fpath.split('_')[1]

        self.labels.append(TUMOR_TYPE_TO_LABEL[tumor_type])

    def _load_feat_indices(self, fpath):
        """

       Returns indices of top 24 features to be selected
        """
        with np.load(os.path.join(fpath, 'feature_ids.npz')) as data:
            indices = data['indices'][:24]
            indices = torch.from_numpy(indices).to(self.device)
            return indices

    def _get_cell_features_dim(self):

        with h5py.File(complete_path(self.cell_graph_path, self.h5_fnames[0]), 'r') as f:
            cell_features = h5_to_tensor(f['instance_features'], self.device)
            centroid = h5_to_tensor(f['instance_centroid_location'], self.device)
            f.close()
        return cell_features.shape[1] + centroid.shape[1]

    def _get_superpx_features_dim(self):

        with h5py.File(complete_path(self.superpx_graph_path, self.h5_fnames[0]), 'r') as f:
            feat = h5_to_tensor(f['sp_features'], self.device)
            centroid = h5_to_tensor(f['sp_centroids'], self.device)
            select_feat = torch.index_select(feat, 1, self.top_feat_ind).to(self.device)
            f.close()
        return select_feat.shape[1] + centroid.shape[1]

    def _build_normaliser(self, graph_type):
        """
        Build normalizers to normalize the node features (ie, mean=0, std=1)
        """
        if not NORMALIZATION_FACTORS[graph_type]:
            vars(self)[graph_type + '_transform'] = compute_normalization_factor(
                self.data_path, self.h5_fnames
            )
        else:
            vars(self)[graph_type + '_transform'] = NORMALIZATION_FACTORS[graph_type]

    def _load_image(self, img_name):
        return load_image(os.path.join(self.image_path, img_name + '.png')), img_name

    def _build_cell_graph(self, index):
        """
        Build the cell graph
        """
        # extract the image size, centroid, cell features and label
        with h5py.File(complete_path(self.cell_graph_path, self.h5_fnames[index]), 'r') as f:
            image_size = h5_to_tensor(f['image_dimension'], self.device)
            cell_features = h5_to_tensor(f['instance_features'], self.device)
            centroid = h5_to_tensor(f['instance_centroid_location'], self.device)
            image_size = image_size.type(torch.float32)
            norm_centroid = centroid / image_size[:-1]
            f.close()

        # normalize the cell features
        cell_features = \
            (cell_features - self.cell_graph_transform['mean'].to(self.device)) / \
            (self.cell_graph_transform['std']).to(self.device)

        # concat spatial + appearance features
        cell_features = torch.cat((cell_features, norm_centroid), dim=1)

        # build topology
        cell_graph = self.cell_graph_builder(cell_features, centroid)

        return cell_graph

    def _build_superpx_graph(self, index):
        """
        Build the super pixel graph
        """
        # extract the image size, centroid, cell features and label
        with h5py.File(complete_path(self.superpx_graph_path, self.h5_fnames[index]), 'r') as f:
            d_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            feat = h5_to_tensor(f['sp_features'], self.device).type(d_type)
            select_feat = torch.index_select(feat, 1, self.top_feat_ind).to(self.device)
            centroid = h5_to_tensor(f['sp_centroids'], self.device).type(d_type)
            # converting centroid coord from [y, x] to [x, y]
            centroid = torch.index_select(centroid, 1, torch.LongTensor([1, 0]).to(self.device)).to(self.device)

            sp_map = h5_to_tensor(f['sp_map'], self.device).type(d_type)
            image_size = torch.FloatTensor(list(sp_map.shape)).to(self.device)
            norm_centroid = centroid / image_size
            f.close()

        # choose indices of norm factors
        norm_mean = torch.index_select(self.superpx_graph_transform['mean'], 0, self.top_feat_ind.cpu())
        norm_stddev = torch.index_select(self.superpx_graph_transform['std'], 0, self.top_feat_ind.cpu())

        # normalize the cell features
        features = \
            (select_feat - norm_mean.to(self.device)) / norm_stddev.to(self.device)

        # concat spatial + appearance features
        features = torch.cat((features, norm_centroid), dim=1).to(torch.float)

        # build topology
        graph_file = self._load_graph(self.h5_fnames[index].replace('.h5', ''))
        superpx_graph = self.superpx_graph_builder(features, centroid, graph_file)

        return superpx_graph

    def _build_assignment_matrix(self, index):

        with h5py.File(complete_path(self.superpx_graph_path, self.h5_fnames[index]), 'r') as f:
            sp_map = h5_to_tensor(f['sp_map'], self.device) - 1   # indexing starts from 0
            f.close()

        with h5py.File(complete_path(self.cell_graph_path, self.h5_fnames[index]), 'r') as f:
            cell_location = torch.floor(h5_to_tensor(f['instance_centroid_location'], self.device)).to(torch.long)
            f.close()

        cell_to_superpx = sp_map[cell_location[:, 1], cell_location[:, 0]].cpu().to(torch.long).numpy()
        assignment_matrix = np.zeros((int(cell_to_superpx.shape[0]), 1 + int(torch.max(sp_map).item())))
        assignment_matrix[np.arange(cell_to_superpx.size), cell_to_superpx] = 1

        return torch.from_numpy(assignment_matrix).float().t().to(self.device)

    def __getitem__(self, index):
        """
        Get an example.

        Args:
            index (int): index of the example.
        Returns: - data (list)
                 - label (int)
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
            assignment_matrix = self._build_assignment_matrix(index)
            data.append(assignment_matrix)

        # 4. load the image if required
        if self.load_image:
            image, image_name = self._load_image(self.h5_fnames[index].replace('.h5', ''))
            data.append(image)
            data.append(image_name)

        return data, label

    def __len__(self):
        """Return the number of samples in the WSI."""
        return self.num_samples

    def get_node_dims(self):
        if hasattr(self, 'num_cell_features') and hasattr(self, 'num_superpx_features'):
            return self.num_cell_features, self.num_superpx_features
        elif hasattr(self, 'num_cell_features'):
            return self.num_cell_features
        elif hasattr(self, 'num_superpx_features'):
            return self.num_superpx_features


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

    data_dir = get_dir_in_folder(complete_path(path, 'nuclei_info'))
    data_dir = list(filter(lambda x: all(b not in x for b in DATASET_BLACKLIST), data_dir))

    datasets = {}

    for data_split in ['train', 'val', 'test']:
        datasets[data_split] = torch.utils.data.ConcatDataset(
            datasets=[
                PascaleDataset(
                    path,
                    dir,
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

        # get the node dimension
        node_dims = data.datasets[0].get_node_dims()

    return dataloaders, node_dims
