"""Pascale Dataset loader."""
import os
import h5py
import torch.utils.data
import numpy as np
from dgl.data.utils import load_graphs
import dgl 

from histocartography.dataloader.base_dataloader import BaseDataset
from histocartography.utils.io import (
    complete_path, h5_to_tensor,
    load_h5_fnames, get_dir_in_folder
)
from histocartography.dataloader.constants import get_tumor_type_to_label
from histocartography.dataloader.constants import NODE_FEATURE_TYPE_TO_DIRNAME, NODE_FEATURE_TYPE_TO_H5, ALL_DATASET_NAMES
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN, GNN_EDGE_FEAT
from histocartography.utils.vector import compute_normalization_factor
from histocartography.utils.io import load_image
from histocartography.utils.graph import set_graph_on_cuda


class CellDataset(BaseDataset):
    """Cell data loader."""

    def __init__(
            self,
            data_path,
            dataset_name,
            split,
            cuda,
            config,
            load_in_ram=False,
    ):
        """
        Pascale dataset constructor.

        Args: 
            :param data_path: (str) path to the data (currently stored on dataT/pus/histocartography)
            :param dataset_name: (str) data are grouped as "datasets" - each of them corresponds to a tumor type 
            :param slit: (str) if we should load the train/test/val split
            :param cuda (bool): cuda usage
            :param config: (dict) config file
            :param load_in_ram: (bool) if we load the data in RAM before -- true for train / false for debugging
        """
        super(CellDataset, self).__init__(config, cuda)

        print('Start loading dataset {} : {}'.format(dataset_name, split))

        # 1. set class attributes
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.load_in_ram = load_in_ram
        self.tumor_type_to_label = get_tumor_type_to_label(class_split)

        # 2. load h5 fnames and labels (from h5 fname)
        self._load_h5_fnames_and_labels(data_path, split)

        # 3. extract meta data
        self.num_samples = len(self.h5_fnames)

        self.drop_cg_appearance_features = config['graph_building']['cell_graph_builder']['drop_appearance_features']
        self.cell_node_feature_types = config['graph_building']['cell_graph_builder']['node_feature_types']
        self.encode_cg_edges = config['graph_building']['cell_graph_builder']['edge_encoding']
        self.base_cell_graph_path = os.path.join(self.data_path, 'graphs', 'cell_graphs')
        self.base_cell_instance_map_path = os.path.join(self.data_path, 'nuclei_info', 'nuclei_detected', 'instance_map')
        self.num_cell_features = self._get_cell_features_dim()
        self.base_nuclei_label_path = os.path.join(self.data_path, '../', 'Nuclei', 'predictions')
        if load_in_ram:
            self._load_cell_graph_in_ram()

    def _load_cell_graph_in_ram(self):
        """
        Load cell graphs in the RAM
        """
        self.cell_graphs = []
        for i in range(self.num_samples):
            self.cell_graphs.append(self._build_cell_graph(i))

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

        # drop edge features
        del g.edata[GNN_EDGE_FEAT]

        # disable read only
        g.readonly(False)

        return g

    def __getitem__(self, index):
        """
        Get an example.

        Args:
            index (int): index of the example.
        Returns: - data (list)
                 - label (int)
        """

        data = []
        if self.load_in_ram:
            cell_graph = self.cell_graphs[index]
        else:
            cell_graph = self._build_cell_graph(index)

        if self.cuda:
            cell_graph = set_graph_on_cuda(cell_graph)
        data.append(cell_graph)

        return data

    def __len__(self):
        """Return the number of samples in the WSI."""
        return self.num_samples

    def get_node_dims(self):
        return self.num_cell_features, self.num_edge_cell_features


def collate(batch):
    """
    Collate a batch.

    Args:
        batch (torch.tensor): a batch of examples.

    Returns:
        data: (tuple)
        labels: (torch.LongTensor)
    """

    data = dgl.batch([example for example in batch])
    return data


def build_datasets(path, class_split, cuda, *args, **kwargs):
    """
    Builds dataset from text files that contain train:test:validation split

    Returns:
        Cell datasets for train, validation and testing
    """

    # data_dir = get_dataset_white_list(class_split)
    data_dir = ["adh",  "benign", "dcis", "fea", "malignant", "pathologicalbenign", "udh"]
    datasets = {}

    for data_split in ['train', 'val', 'test']:
        datasets[data_split] = torch.utils.data.ConcatDataset(
            datasets=[
                CellDataset(
                    path,
                    dir,
                    data_split,
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
                shuffle=(split == 'train'),
                num_workers=num_workers,
                collate_fn=collate
            )

        # get the node dimension
        node_dims = data.datasets[0].get_node_dims()

    return dataloaders, node_dims
