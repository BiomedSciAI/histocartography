"""
Module for loading data and building graph for superpx
"""
import os
import glob
import h5py
import numpy as np
import importlib
import torch
from tqdm import tqdm 

from histocartography.graph_generation.constants import AVAILABLE_GRAPH_BUILDERS
from histocartography.dataloader.constants import NORMALIZATION_FACTORS
from histocartography.utils.io import read_params, h5_to_tensor
from histocartography.graph_generation.utils.utils_build import feature_normalization, build_graph, save_graph_file


class BuildSPGraph(object):

    def __init__(
            self,
            args,
            device):

        # setting class attributes
        self.args = args
        self.config = read_params(args.configuration, verbose=True)
        self.data_path = self.args.data_path
        self.model_type = self.config['model_type']
        self.features_used = self.args.features_used
        self.device = device
        self.save_path = self.args.save_path

        # Build graph normalizer
        self._build_normaliser(graph_type=self.model_type)

        # Initialise graph builder
        for graph_type, param in self.config['graph_building'].items():
            self._construct_graph_builder(param, name=graph_type)

    def run(self):
        self._build_superpx_graph(os.path.join(self.data_path, 'super_pixel_info'), self.features_used)

    def _build_normaliser(self, graph_type):
        """
        Build normalizers to normalize the node features (ie, mean=0, std=1)
        """
        try:
            vars(self)[graph_type + '_transform'] = NORMALIZATION_FACTORS['features_hc_'][graph_type]
        except:
            print('No normalization')
       
    def _construct_graph_builder(self, config, name):

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

    def _load_feat_indices(self, fpath):
        """

       Returns indices of top 24 features to be selected
        """
        with np.load(os.path.join(fpath, 'feature_ids.npz')) as data:
            indices = data['indices'][:24]
            indices = torch.from_numpy(indices).to(self.device)
            return indices

    def _build_superpx_graph(self, path_to_superpx, features_type):

        path_to_centroid_global = os.path.join(path_to_superpx, 'sp_merged_detected', 'merging_hc', 'centroids')
        path_to_features_global = os.path.join(path_to_superpx, 'sp_merged_features', features_type)
        path_to_map_global = os.path.join(path_to_superpx, 'sp_merged_detected', 'merging_hc', 'instance_map')
        path_to_feature_indices = os.path.join(self.data_path, 'misc_utils', 'merging_sp_classification', 'sp_classifier')

        self.top_feat_ind = self._load_feat_indices(path_to_feature_indices)
        tumor_types = [name for name in os.listdir(path_to_centroid_global) if
                       os.path.isdir(os.path.join(path_to_centroid_global, name))]
        for tumor_type in tumor_types:
            centroid_file_path = os.path.join(path_to_centroid_global, tumor_type)
            feature_file_path = os.path.join(path_to_features_global, tumor_type)
            map_file_path = os.path.join(path_to_map_global, tumor_type)
            file_list_centroid = glob.glob('%s/*.h5' % centroid_file_path)
            for centroid_file in tqdm(file_list_centroid):
                # get basename to use for loading features and centroid data
                cent_filename = os.path.basename(centroid_file)
                basename = cent_filename.split('.')[0]

                # read centroid data
                with h5py.File(os.path.join(centroid_file_path, cent_filename), 'r') as f:
                    superpx_centroid = h5_to_tensor(f['instance_centroid_location'], self.device)
                    image_size = h5_to_tensor(f['image_dimension'], self.device)
                    f.close()

                # read features
                with h5py.File(os.path.join(feature_file_path, cent_filename), 'r') as g:
                    superpx_features = h5_to_tensor(g['embeddings'], self.device)
                    g.close()

                # read instance map
                with h5py.File(os.path.join(map_file_path, cent_filename), 'r') as h:
                    superpx_map = h5_to_tensor(h['detected_instance_map'], self.device)
                    h.close()

                # normalize & select features (only with handcrafted features)
                if features_type == 'merging_hc_features_hc_':
                    norm_mean = torch.index_select(self.superpx_graph_model_transform['mean'], 0,
                                                   self.top_feat_ind.cpu())
                    norm_stddev = torch.index_select(self.superpx_graph_model_transform['std'], 0,
                                                     self.top_feat_ind.cpu())
                    superpx_features = torch.index_select(superpx_features, 1, self.top_feat_ind).to(self.device)
                    superpx_features = \
                        (superpx_features - norm_mean.to(self.device)) / norm_stddev.to(self.device)

                image_dim = image_size.type(torch.float32)
                image_dim = torch.index_select(image_dim, 0, torch.LongTensor([1, 0]).to(self.device)).to(self.device)
                image_dim = image_dim.type(torch.float32)
                norm_centroid = superpx_centroid / image_dim

                # concat spatial + appearance features
                features = torch.cat((superpx_features, norm_centroid), dim=1).to(torch.float)

                # build graph
                superpx_graph = build_graph(self.superpx_graph_builder, self.config, features, superpx_centroid,
                                             superpx_map)

                # save graph (original)
                save_graph_file(self.save_path, tumor_type, self.config, features_type, cent_filename, superpx_graph)
