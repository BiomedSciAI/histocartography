from sklearn.model_selection import ParameterGrid

from histocartography.ml.models.constants import AVAILABLE_MODEL_TYPES
from histocartography.utils.io import write_json, complete_path, check_for_dir
from histocartography.dataloader.constants import get_tumor_type_to_label, get_dataset_black_list
import numpy as np


MODEL_TYPE_TO_MODEL_PARAMS = {
    'multi_level_graph_model': '_get_multi_level_graph_model_params',
    'cell_graph_model': '_get_cell_graph_model_params',
    'superpx_graph_model': '_get_superpx_graph_model_params',
    'concat_graph_model': '_get_concat_graph_model_params'
}

MODEL_TYPE_TO_GRAPH_BUILDING_PARAMS = {
    'multi_level_graph_model': '_get_superpx_cell_graph_building_params',
    'cell_graph_model': '_get_cell_graph_building_params',
    'superpx_graph_model': '_get_superpx_graph_building_params',
    'concat_graph_model': '_get_superpx_cell_graph_building_params'
}


class ConfigGenerator:

    def __init__(self, save_path, num_classes=5, gnn_layer_type='gin_layer'):
        """
        Config Generator constructor
        """

        self.save_path = save_path
        self.tumor_type_to_label = get_tumor_type_to_label(num_classes)
        self.dataset_blacklist = get_dataset_black_list(num_classes)
        self.gnn_layer_type = gnn_layer_type

    def __call__(self, model_type):
        """
        Call config generator
        :return:
        """

        if model_type not in AVAILABLE_MODEL_TYPES.keys():
            raise ValueError("Unrecognised model type: {}. Please provide one from: {}".format(
                model_type,
                list(AVAILABLE_MODEL_TYPES.keys())
            ))

        grid_params = ParameterGrid(
            {
                'graph_building': getattr(self, MODEL_TYPE_TO_GRAPH_BUILDING_PARAMS[model_type])(),
                'model_params': getattr(self, MODEL_TYPE_TO_MODEL_PARAMS[model_type])(),
                'model_type': [model_type]
            }
        )

        for i, params in enumerate(grid_params):
            fname = model_type + '_config_' + str(i) + '.json'
            save_path = complete_path(self.save_path, model_type + '_config')
            check_for_dir(save_path)
            write_json(complete_path(save_path, fname), params)

    def _get_cell_graph_model_params(self):
        config = self._get_base_model_params()
        config.param_grid[0]['gnn_params'] = self._get_cell_gnn_params()
        config.param_grid[0]['readout'] = self._get_readout_params()
        return config

    def _get_multi_level_graph_model_params(self):
        config = self._get_base_model_params()
        config.param_grid[0]['gnn_params'] = self._get_superpx_cell_gnn_params()
        config.param_grid[0]['readout'] = self._get_readout_params()
        return config

    def _get_superpx_graph_model_params(self):
        config = self._get_base_model_params()
        config.param_grid[0]['gnn_params'] = self._get_superpx_gnn_params()
        config.param_grid[0]['readout'] = self._get_readout_params()
        return config

    def _get_concat_graph_model_params(self):
        config = self._get_base_model_params()
        config.param_grid[0]['gnn_params'] = self._get_superpx_cell_gnn_params()
        config.param_grid[0]['readout'] = self._get_readout_params()
        return config

    def _get_cell_gnn_params(self):
        config = ParameterGrid(
            {
                "cell_gnn": self._get_pna_params()
            }
        )
        return config

    def _get_superpx_cell_gnn_params(self):
        config = ParameterGrid(
            {
                "cell_gnn": self._get_pna_params(),
                "superpx_gnn": self._get_pna_params()
            }
        )
        return config

    def _get_superpx_gnn_params(self):
        config = ParameterGrid(
            {
                "superpx_gnn": self._get_pna_params()
            }
        )
        return config

    def _get_cell_graph_building_params(self):
        config = ParameterGrid(
            {
                "cell_graph_builder": self._get_knn_graph_building_params()
            }
        )
        return config

    def _get_superpx_graph_building_params(self):
        config = ParameterGrid(
            {
                "superpx_graph_builder": self._get_rag_graph_building_params()
            }
        )
        return config

    def _get_superpx_cell_graph_building_params(self):
        config = ParameterGrid(
            {
                "superpx_graph_builder": self._get_rag_graph_building_params(),
                "cell_graph_builder": self._get_knn_graph_building_params()
            }
        )
        return config

    def _get_base_model_params(self):
        config = ParameterGrid(
            {
                "dropout": [0.0],
                "num_classes": self._get_number_classes(self.dataset_blacklist, self.tumor_type_to_label),
                "use_bn": [True],
                "activation": ["relu"]
            }
        )
        return config

    @staticmethod
    def _get_number_classes(blacklist, labels):
        for i in range(len(blacklist)):
            # del labels[blacklist[i]]
            labels.pop(blacklist[i], None)
        n_classes = len(np.unique(list(labels.values())))
        return [n_classes]

    def _get_gnn_params(self):
        if self.gnn_layer_type == 'dense_gin_layer':
            gnn_config = self._get_dense_gin_params()
        elif self.gnn_layer_type == 'gin_layer':
            gnn_config = self._get_gin_params()
        else:
            raise NotImplementedError('Other GNN layers not implemented.')
        return gnn_config

    @staticmethod
    def _get_gin_params():

        config = ParameterGrid(
            {
                "layer_type": ["gin_layer"],
                "activation": ["relu"],
                "n_layers": [4],  # [3, 4, 5]
                "neighbor_pooling_type": ["mean"],
                "hidden_dim": [64],
                "output_dim": [64],
                "graph_norm": [True],
                "agg_operator": ["lstm"]  # "concat"
            }
        )

        return config

    @staticmethod
    def _get_pna_params():

        config = ParameterGrid(
            {
                "layer_type": ["pna_layer"],
                "activation": ["relu"],
                "n_layers": [4],
                "hidden_dim": [64],
                "output_dim": [64],
                "agg_operator": ["lstm", "concat"],
                "residual": [True],
                "graph_norm": [True],
                "aggregators": ["mean max min std"],
                "scalers": ["identity amplification attenuation"],
                "towers": [1],
                "divide_input_first": [True],
                "pretrans_layers" : [1],
                "posttrans_layers" : [1]

            }
        )

        return config


    @staticmethod
    def _get_dense_gin_params():

        config = ParameterGrid(
            {
                "layer_type": ["dense_gin_layer"],
                "activation": ["relu"],
                "n_layers": [3, 4, 5],
                "neighbor_pooling_type": ["mean"],
                "hidden_dim": [32],
                "output_dim": [32]
            }
        )

        return config

    @staticmethod
    def _get_explainer_params():

        config = ParameterGrid(
            {
                "loss": {
                    "adj": [0.005],
                    "adj_ent": [1.0],
                    "node_ent": [1.0],
                    "node": [0.05],
                    "ce": [10.0]
                },
                "adj_thresh": [0.1],
                "node_thresh": [0.1],
                "init": ["normal"],
                "mask_activation": ["sigmoid"]
            }
        )

        return config

    @staticmethod
    def _get_gat_params():

        config = ParameterGrid(
            {
                "activation": ["relu"],
                "hidden_dim": [64],
                "layer_type": ["gat_layer"],
                "n_layers": [2],
                "output_dim": [64],
                "feat_drop": [0.0],
                "attn_drop": [0.0],
                "negative_slope": [0.2],
                "residual": [False],
                "num_heads": [2]
            }
        )

        return config

    @staticmethod
    def _get_readout_params():
        config = ParameterGrid(
            {
                "num_layers": [2],
                "hidden_dim": [128]
            }
        )
        return config

    @staticmethod
    def _get_knn_graph_building_params():
        config = ParameterGrid(
            {
                "graph_building_type": ["knn_graph_builder"],
                "n_neighbors": [5],
                "max_distance": [50],
                "edge_encoding": [False],
                'node_feature_types': [
                    # ['features_cnn_resnet101_mask_False_', 'centroid'], 
                    #['features_cnn_resnet50_mask_True_'], 
                    #['features_cnn_resnet34_mask_True_'], 
                    # ['features_cnn_vgg16_mask_False_', 'centroid'], 
                    # ['features_cnn_vgg19_mask_False_', 'centroid'], 
                    # ['features_hc_', 'centroid'], 
                    # ['features_cnn_resnet101_mask_True_', 'centroid'], 
                    #['features_cnn_resnet50_mask_False_'], 
                    ['features_cnn_resnet34_mask_False_'], 
                    # ['features_cnn_vgg16_mask_True_', 'centroid'], 
                    # ['features_cnn_vgg19_mask_True_', 'centroid'], 
                    # ['nuclei_vae_features', 'centroid']
                ]
            }
        )
        return config

    @staticmethod
    def _get_rag_graph_building_params():
        config = ParameterGrid(
            {
                "graph_building_type": ["rag_graph_builder"],
                "edge_encoding": [False],
                'node_feature_types': [
                    ['merging_hc_features_cnn_resnet34_mask_False_'],
                    #['merging_hc_features_cnn_resnet50_mask_False_'],
                    #['merging_hc_features_cnn_resnet34_mask_True_'],
                    #['merging_hc_features_cnn_resnet50_mask_True_']
                ]
            }
        )
        return config
