from sklearn.model_selection import ParameterGrid

from histocartography.ml.models.constants import AVAILABLE_MODEL_TYPES
from histocartography.utils.io import write_json, complete_path, check_for_dir


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

    def __init__(self, save_path):
        """
        Config Generator constructor
        """

        self.save_path = save_path

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
            write_json(params, complete_path(save_path, fname))

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
                "cell_gnn": self._get_gnn_params()
            }
        )
        return config

    def _get_superpx_cell_gnn_params(self):
        config = ParameterGrid(
            {
                "cell_gnn": self._get_gnn_params(),
                "superpx_gnn": self._get_gnn_params()
            }
        )
        return config

    def _get_superpx_gnn_params(self):
        config = ParameterGrid(
            {
                "superpx_gnn": self._get_gnn_params()
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
                "superpx_graph_builder": self._get_knn_graph_building_params()
            }
        )
        return config

    def _get_superpx_cell_graph_building_params(self):
        config = ParameterGrid(
            {
                "superpx_graph_builder": self._get_knn_graph_building_params(),
                "cell_graph_builder": self._get_knn_graph_building_params()
            }
        )
        return config

    def _get_gnn_params(self):
        gnn_config = self._get_gin_params()
        gat_config = self._get_gat_params()
        gnn_config.param_grid.append(gat_config.param_grid[0])
        return gnn_config

    @staticmethod
    def _get_gin_params():

        config = ParameterGrid(
            {
                "layer_type": ["gin_layer"],
                "activation": ["relu"],
                "n_layers": [2, 3],
                "neighbor_pooling_type": ["mean", "sum"],
                "hidden_dim": [32],
                "output_dim": [32]
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
                "hidden_dim": [64]
            }
        )
        return config

    @staticmethod
    def _get_base_model_params():
        config = ParameterGrid(
            {
                "dropout": [0.0],
                "num_classes": [2],
                "use_bn": [False, True],
                "cat": [False, True],
                "activation": ["relu"]
            }
        )
        return config

    @staticmethod
    def _get_knn_graph_building_params():
        config = ParameterGrid(
            {
                "graph_building_type": ["knn_graph_builder"],
                "n_neighbors": [5],
                "max_distance": [50, 100],
                "edge_encoding": [False]
            }
        )
        return config

    @staticmethod
    def _get_rag_graph_building_params():
        config = ParameterGrid(
            {
                "graph_building_type": ["rag_graph_builder"]
            }
        )
        return config
