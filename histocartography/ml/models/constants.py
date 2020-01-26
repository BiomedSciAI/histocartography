AVAILABLE_MODEL_TYPES = {
    'multi_level_graph_model': 'MultiLevelGraphModel',
    'cell_graph_model': 'CellGraphModel',
    'superpx_graph_model': 'SuperpxGraphModel'
}

MODEL_TYPE = 'model_type'

MODEL_MODULE = 'histocartography.ml.models.{}'


def load_cell_graph(model_type):
    return model_type == 'cell_graph_model' or model_type == 'multi_level_graph_model'


def load_superpx_graph(model_type):
    return model_type == 'superpx_graph_model' or model_type == 'multi_level_graph_model'
