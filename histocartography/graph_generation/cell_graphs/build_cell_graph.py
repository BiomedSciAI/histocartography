"""
Module for loading cell data and building graph
"""
import os
import glob
import h5py
import numpy as np
import importlib
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm 

from histocartography.graph_generation.constants import AVAILABLE_GRAPH_BUILDERS
from histocartography.dataloader.constants import NORMALIZATION_FACTORS
from histocartography.utils.io import read_params, h5_to_tensor
from histocartography.graph_generation.utils.utils_build import feature_normalization, build_graph, \
    save_graph_file, get_data_aug


class BuildCellGraph(object):

    def __init__(
            self,
            args, 
            device):

        # setting class attributes

        self.args = args
        self.device = device
        self.config = read_params(args.configuration, verbose=True)
        self.data_path = self.args.data_path
        self.model_type = self.config['model_type']
        self.features_used = self.args.features_used
        self.save_path = self.args.save_path

        # Build graph normalizer
        self._build_normaliser(graph_type=self.model_type)

        # Initialise graph builder
        for graph_type, param in self.config['graph_building'].items():
            self._construct_graph_builder(param, name=graph_type)

    def run(self):
        self._build_cell_graph(os.path.join(self.data_path, 'nuclei_info'), self.features_used)

    def _build_normaliser(self, graph_type):
        """
        Build normalizers to normalize the node features (ie, mean=0, std=1)
        """
        if False and self.features_used in NORMALIZATION_FACTORS[graph_type].keys():  # by-pass the normalization part for now 
            vars(self)[graph_type + '_transform'] = NORMALIZATION_FACTORS[graph_type][self.features_used]

    def _construct_graph_builder(self, config, name):
        """
        Build graph builder
        """

        graph_builder_type = config['graph_building_type']
        graph_building_module = 'graph_builders.{}'
        if graph_builder_type in list(AVAILABLE_GRAPH_BUILDERS.keys()):
            module = importlib.import_module(
                graph_building_module.format(graph_builder_type)
            )
            vars(self)[name] = getattr(
                module, AVAILABLE_GRAPH_BUILDERS[graph_builder_type])(config)
        else:
            raise ValueError(
                'Graph builder type: {} not recognized. Options are: {}'.format(
                    graph_builder_type, list(
                        AVAILABLE_GRAPH_BUILDERS.keys()
                    )
                )
            )

    def _build_cell_graph(self, path_to_nuclei, features_type):
        """

        :param path_to_nuclei: path to cell graph data
        :param features_type: hand-crafted features or CNN feats
        :return: returns cell graph and saves it
        """

        path_to_centroid_global = os.path.join(path_to_nuclei, 'nuclei_detected', 'centroids')
        path_to_instance_map_global = os.path.join(path_to_nuclei, 'nuclei_detected', 'instance_map')
        path_to_features_global = os.path.join(path_to_nuclei, 'nuclei_features', features_type)

        tumor_types = [name for name in os.listdir(path_to_centroid_global) if os.path.isdir(os.path.join(path_to_centroid_global, name))]

        for tumor_type in tumor_types:
            centroid_file_path = os.path.join(path_to_centroid_global, tumor_type)
            feature_file_path = os.path.join(path_to_features_global, tumor_type)
            instance_map_path = os.path.join(path_to_instance_map_global, tumor_type)

            file_list_centroid = glob.glob('%s/*.h5' % centroid_file_path)

            print('Start processing tumor type', tumor_type)
            for centroid_file in tqdm(file_list_centroid):
                # get basename to use for loading features and centroid data
                cent_filename = os.path.basename(centroid_file)
                basename = cent_filename.split('.')[0]

                # read centroid data
                with h5py.File(os.path.join(centroid_file_path, cent_filename), 'r') as f:
                    cell_centroid = h5_to_tensor(f['instance_centroid_location'],self.device)
                    image_size = h5_to_tensor(f['image_dimension'], self.device)
                    f.close()

                # read features
                with h5py.File(os.path.join(feature_file_path, cent_filename), 'r') as f:
                    cell_features = h5_to_tensor(f['embeddings'], self.device)
                    f.close()

                # read instance map
                with h5py.File(os.path.join(instance_map_path, '_h5', cent_filename), 'r') as f:
                    instance_map = h5_to_tensor(f['detected_instance_map'], self.device)
                    f.close()

                # feature normalisation for original cell features
                cell_features = feature_normalization(cell_centroid, cell_features, image_size,
                                                      None, self.device)

                # build graph
                cell_graph = build_graph(self.cell_graph_builder, self.config, cell_features, cell_centroid, instance_map)

                # save graph
                save_graph_file(self.save_path, tumor_type, self.config, features_type, cent_filename, cell_graph)
