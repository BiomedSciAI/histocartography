"""
Module for building assignment matrix for HACT
"""
import os
import glob
import h5py
import numpy as np
import torch 
import importlib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from histocartography.graph_generation.constants import AVAILABLE_GRAPH_BUILDERS, NORMALIZATION_FACTORS
from histocartography.utils.io import read_params, h5_to_tensor
from histocartography.graph_generation.utils.utils_build import feature_normalization, build_graph, \
    save_mat, get_data_aug


class BuildAssignmnentMatrix(object):

    def __init__(
            self,
            args,
            device):

        # setting class attributes

        self.args = args
        self.config = read_params(args.configuration, verbose=True)
        self.data_path = self.args.data_path
        self.model_type = self.config['model_type']
        self.save_path = self.args.save_path
        self.device = device


    def run(self):
        nuclei_path = os.path.join(self.data_path, 'nuclei_info')
        superpx_path = os.path.join(self.data_path, 'super_pixel_info')
        self._build_assignment_matrix(nuclei_path, superpx_path)


    def _build_assignment_matrix(self, path_to_nuclei, path_to_superpx):
        """

        :param path_to_nuclei: path to cell graph data
        :param path_to_super_pixel: path to superpixel graph data
        :return: returns cell graph and saves it
        """

        # Centroids of cell graphs

        path_to_centroid_global = os.path.join(path_to_nuclei, 'nuclei_detected', 'centroids')
        path_to_sp_map_global = os.path.join(path_to_superpx, 'sp_merged_detected', 'merging_hc', 'instance_map')

        tumor_types = [name for name in os.listdir(path_to_centroid_global) if os.path.isdir(os.path.join(path_to_centroid_global,name))]

        for tumor_type in tumor_types:
            centroid_file_path = os.path.join(path_to_centroid_global, tumor_type)
            sp_map_path = os.path.join(path_to_sp_map_global, tumor_type)
            file_list_centroid = glob.glob('%s/*.h5' % centroid_file_path)
            file_list_sp_map = glob.glob('%s/*.h5' % sp_map_path)

            print('Start processing tumor type', tumor_type)
            
            for centroid_file in tqdm(file_list_centroid):
                # get basename to use for loading features and centroid data
                cent_filename = os.path.basename(centroid_file)
                basename = cent_filename.split('.')[0]

                # read centroid data
                with h5py.File(os.path.join(centroid_file_path, cent_filename), 'r') as f:
                    cell_centroid = h5_to_tensor(f['instance_centroid_location'], self.device).to(torch.int64).cpu().detach().numpy()
                    f.close()

                # read super-pixel map data
                with h5py.File(os.path.join(sp_map_path, cent_filename), 'r') as f:
                    sp_map = h5_to_tensor(f['detected_instance_map'], self.device) -1 
                    f.close()

                cell_to_superpx = sp_map[cell_centroid[:, 1], cell_centroid[:, 0]].to(torch.int64).cpu().detach().numpy()
                assignment_matrix = np.zeros((int(cell_to_superpx.shape[0]), 1 + int(torch.max(sp_map).item())))
                assignment_matrix[np.arange(cell_to_superpx.size), cell_to_superpx] = 1

                # save mat
                save_mat(self.save_path, tumor_type, self.config, cent_filename, assignment_matrix)
