"""Pascale Dataset loader."""
import os
import h5py
import torch.utils.data
import numpy as np
from dgl.data.utils import load_graphs

from histocartography.dataloader.base_dataloader import BaseDataset
from histocartography.utils.io import (
    complete_path, h5_to_tensor,
    load_h5_fnames, get_dir_in_folder
)
from histocartography.dataloader.constants import (
    NORMALIZATION_FACTORS, COLLATE_FN)
from histocartography.dataloader.constants import get_tumor_type_to_label
from histocartography.dataloader.constants import get_dataset_white_list
from histocartography.dataloader.constants import NODE_FEATURE_TYPE_TO_DIRNAME, NODE_FEATURE_TYPE_TO_H5, ALL_DATASET_NAMES
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN, GNN_EDGE_FEAT
from histocartography.utils.vector import compute_normalization_factor
from histocartography.utils.io import load_image
from histocartography.utils.graph import set_graph_on_cuda


class PascaleDataset(BaseDataset):
    """Pascale data loader."""

    def __init__(
            self,
            data_path,
            dataset_name,
            split,
            class_split,
            cuda,
            config,
            load_cell_graph=True,
            load_superpx_graph=True,
            load_image=False,
            load_in_ram=False,
            show_superpx=False,
            fold_id=None
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
        self.show_superpx = show_superpx
        self.load_in_ram = load_in_ram
        self.fold_id = fold_id
        self.tumor_type_to_label = get_tumor_type_to_label(class_split)

        # 2. load h5 fnames and labels (from h5 fname)
        self._load_h5_fnames_and_labels(data_path, split)

        # 3. extract meta data
        self.num_samples = len(self.h5_fnames)

        if load_cell_graph:
            self.drop_cg_appearance_features = config['graph_building']['cell_graph_builder']['drop_appearance_features']
            self.cell_node_feature_types = config['graph_building']['cell_graph_builder']['node_feature_types']
            self.encode_cg_edges = config['graph_building']['cell_graph_builder']['edge_encoding']
            self.base_cell_graph_path = os.path.join(self.data_path, 'graphs', 'cell_graphs')
            self.num_cell_features = self._get_cell_features_dim()
            self.num_edge_cell_features = self._get_edge_cell_features_dim()
            if load_in_ram:
                self._load_cell_graph_in_ram()

        if load_superpx_graph:
            self.drop_tg_appearance_features = config['graph_building']['superpx_graph_builder']['drop_appearance_features']
            self.superpx_node_feature_types = config['graph_building']['superpx_graph_builder']['node_feature_types']
            self.encode_tg_edges = config['graph_building']['superpx_graph_builder']['edge_encoding']
            self.base_superpx_graph_path = os.path.join(self.data_path, 'graphs', 'tissue_graphs')
            self.base_superpx_h5_path = os.path.join(self.data_path, 'super_pixel_info')
            self.num_superpx_features = self._get_superpx_features_dim()
            self.num_edge_superpx_features = self._get_edge_superpx_features_dim()
            if load_in_ram:
                self._load_superpx_graph_in_ram()

        if load_cell_graph and load_superpx_graph:
            self.base_assignment_matrix_path = os.path.join(self.data_path, 'assignment_mat')
            if load_in_ram:
                self._load_assignment_matrices_in_ram()

        if load_image:
            self.image_path = os.path.join(
                self.data_path, 'Images_norm', self.dataset_name
            )

    def _load_assignment_matrices_in_ram(self):
        """
        Load assignment matrices in the RAM
        """
        self.assignment_matrices = []
        for i in range(self.num_samples):
            self.assignment_matrices.append(self._build_assignment_matrix(i))

    def _load_cell_graph_in_ram(self):
        """
        Load cell graphs in the RAM
        """
        self.cell_graphs = []
        for i in range(self.num_samples):
            self.cell_graphs.append(self._build_cell_graph(i))

    def _load_superpx_graph_in_ram(self):
        """
        Load superpx graphs in the RAM
        """
        self.superpx_graphs = []
        for i in range(self.num_samples):
            self.superpx_graphs.append(self._build_superpx_graph(i))

    def _load_h5_fnames_and_labels(self, data_path, split):
        """
        Load the h5 data from the text files
        """
        self.labels = []
        extension = '.h5'
        tumor_type = self.dataset_name
        self.h5_fnames = load_h5_fnames(data_path, tumor_type, extension, split, self.fold_id)

        for fname in self.h5_fnames:
            self._load_label(fname)

    def _load_label(self, fpath):
        """
        Load the label by inspecting the filename
        """
        tumor_type = fpath.split('_')[1]
        try:
            label = self.tumor_type_to_label[tumor_type]
        except:
            print('Warning: unable to read label')
            label = 0
        self.labels.append(label)

    def _get_cell_features_dim(self):
        try:
            if self.drop_cg_appearance_features:
                dim = 2
            else:
                graph_fname = os.path.join(
                    self.base_cell_graph_path,
                    self.cell_node_feature_types[0],
                    self.dataset_name,
                    self.h5_fnames[0].replace('.h5', '.bin')
                )
                g, _ = load_graphs(graph_fname)
                dim = g[0].ndata[GNN_NODE_FEAT_IN].shape[1]
        except:
            print('Warning: List of DGL graphs is empty. Tentative dimension set to 2050.')
            dim = 2050  # corresponds to resnet50 + location embeddings 
        return dim

    def _get_edge_cell_features_dim(self):
        return 4 if self.encode_cg_edges else None 

    def _get_superpx_features_dim(self):
        try:
            if self.drop_tg_appearance_features:
                dim = 2
            else:
                graph_fname = os.path.join(
                    self.base_superpx_graph_path,
                    self.superpx_node_feature_types[0],
                    self.dataset_name,
                    self.h5_fnames[0].replace('.h5', '.bin')
                )
                g, _ = load_graphs(graph_fname)
                dim = g[0].ndata[GNN_NODE_FEAT_IN].shape[1]
        except:
            print('Warning: List of DGL graphs is empty. Tentative dimension set to 2050.')
            dim = 2050  # corresponds to resnet50 + location embeddings 
        return dim

    def _get_edge_superpx_features_dim(self):
        return 4 if self.encode_tg_edges else None 

    def _get_superpx_map(self, index):
        with h5py.File(complete_path(self.superpx_graph_path, self.h5_fnames[index]), 'r') as f:
            d_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            sp_map = h5_to_tensor(f['sp_map'], self.device).type(d_type)
            f.close()
        return sp_map

    def _load_image(self, img_name):
        return load_image(os.path.join(self.image_path, img_name + '.png')), img_name

    def _build_cell_graph(self, index):
        """
        Build the cell graph
        """

        graph_fname = os.path.join(
            self.base_cell_graph_path,
            self.cell_node_feature_types[0],
            self.dataset_name,
            self.h5_fnames[index].replace('.h5', '.bin')
        )
        g, _ = load_graphs(graph_fname)
        g = g[0]

        # keep/drop edge features 
        if not self.encode_cg_edges and GNN_EDGE_FEAT in g.edata.keys():
            del g.edata[GNN_EDGE_FEAT]

        # keep/drop appearance features 
        if self.drop_cg_appearance_features:
            subfeats = g.ndata.pop(GNN_NODE_FEAT_IN)[:, -2:]
            g.ndata[GNN_NODE_FEAT_IN] = subfeats

        return g

    def _build_superpx_graph(self, index):
        """
        Build the super pixel graph
        """
        graph_fname = os.path.join(
            self.base_superpx_graph_path,
            self.superpx_node_feature_types[0],
            self.dataset_name,
            self.h5_fnames[index].replace('.h5', '.bin')
        )
        g, _ = load_graphs(graph_fname)
        g = g[0]

        # keep/drop appearance features 
        if not self.encode_tg_edges and GNN_EDGE_FEAT in g.edata.keys():
            del g.edata[GNN_EDGE_FEAT]

        # keep/drop appearance features 
        if self.drop_tg_appearance_features:
            subfeats = g.ndata.pop(GNN_NODE_FEAT_IN)[:, -2:]
            g.ndata[GNN_NODE_FEAT_IN] = subfeats

        return g

    def _build_assignment_matrix(self, index):
        data = np.load(
            os.path.join(
                self.base_assignment_matrix_path,
                self.dataset_name,
                self.h5_fnames[index].replace('.h5', '.npy')
                )
            )
        return torch.FloatTensor(data).t()
        
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
            if self.load_in_ram:
                cell_graph = self.cell_graphs[index]
            else:
                cell_graph = self._build_cell_graph(index)
            if self.cuda:
                cell_graph = set_graph_on_cuda(cell_graph)
            data.append(cell_graph)

        # 3. load superpx graph
        if self.load_superpx_graph:
            if self.load_in_ram:
                superpx_graph = self.superpx_graphs[index]
            else:
                superpx_graph = self._build_superpx_graph(index)
            if self.cuda:
                superpx_graph = set_graph_on_cuda(superpx_graph)
            data.append(superpx_graph)

        # 5. load superpx map for viz
        if self.show_superpx:
            superpx_map = self._get_superpx_map(index)
            data.append(superpx_map)

        # 4. load assignment matrix to go from the cell graph to the the superpx graph
        if self.load_cell_graph and self.load_superpx_graph:
            if self.load_in_ram:
                assignment_matrix = self.assignment_matrices[index]
            else:
                assignment_matrix = self._build_assignment_matrix(index)
            if self.cuda:
                assignment_matrix = assignment_matrix.cuda()
            data.append(assignment_matrix)

        # 6. load the image if required
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
            return self.num_cell_features, self.num_superpx_features, self.num_edge_cell_features, self.num_edge_superpx_features
        elif hasattr(self, 'num_cell_features'):
            return self.num_cell_features, self.num_edge_cell_features
        elif hasattr(self, 'num_superpx_features'):
            return self.num_superpx_features, self.num_edge_superpx_features


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


def build_datasets(path, class_split, cuda, *args, **kwargs):
    """
    Builds dataset from text files that contain train:test:validation split

    Returns:
        PASCALE datasets for train, validation and testing
    """

    # data_dir = get_dataset_white_list(class_split)
    data_dir = get_dataset_white_list(class_split)
    datasets = {}

    for data_split in ['train', 'val', 'test']:
        datasets[data_split] = torch.utils.data.ConcatDataset(
            datasets=[
                PascaleDataset(
                    path,
                    dir,
                    data_split,
                    class_split,
                    cuda,
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
                shuffle=split == 'train',
                num_workers=num_workers,
                collate_fn=collate
            )

        # get the node dimension
        node_dims = data.datasets[0].get_node_dims()

    return dataloaders, node_dims
