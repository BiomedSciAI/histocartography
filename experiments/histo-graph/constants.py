import networkx as nx
from torchvision.transforms.functional import normalize, resize, to_tensor


# define all the available modules for interpretability 
AVAILABLE_EXPLAINABILITY_METHODS = {
    'GraphPruningExplainer': 'pruning_explainer.graph_pruning_explainer',
    'LRPGNNExplainer': 'lrp_explainer.lrp_gnn_explainer',
    'GraphGradCAMExplainer': 'grad_cam',
    'GraphGradCAMPPExplainer': 'grad_cam'
}



BASE_S3 = 's3://mlflow/'


# used in io.py 
MODEL_TO_MLFLOW_ID = {
    '5_class_scenario': {
        'cell_graph_model': {
            'GNNExplainer': BASE_S3 + '7ad2792ad69940e0a54dc554af3c4716/artifacts/model_best_val_weighted_f1_score_0',   # '55ebbd5a765e4a1d80818112212e1875/artifacts/model_best_val_loss_0',
            'LRPGNNExplainer': BASE_S3 + '7ad2792ad69940e0a54dc554af3c4716/artifacts/model_best_val_weighted_f1_score_0',
            'GraphGradCAMPPExplainer': BASE_S3 + '7ad2792ad69940e0a54dc554af3c4716/artifacts/model_best_val_weighted_f1_score_0',
            'GraphGradCAMExplainer': BASE_S3 + '7ad2792ad69940e0a54dc554af3c4716/artifacts/model_best_val_weighted_f1_score_0'
        },
        'superpx_graph_model': {
            'LRPGNNExplainer': BASE_S3 + '96343b43e4284334910c8901258262d4/artifacts/model_best_val_weighted_f1_score_0',
            'GraphGradCAMPPExplainer': BASE_S3 + '96343b43e4284334910c8901258262d4/artifacts/model_best_val_weighted_f1_score_0',
            'GraphGradCAMExplainer': BASE_S3 + '96343b43e4284334910c8901258262d4/artifacts/model_best_val_weighted_f1_score_0',
            'GNNExplainer': BASE_S3 + '96343b43e4284334910c8901258262d4/artifacts/model_best_val_weighted_f1_score_0'
        }
    },
    '3_class_scenario': {
        'cell_graph_model': {
            'GNNExplainer': BASE_S3 + '29b7f5ee991e4a3e8b553b49a1c3c05a/artifacts/model_best_val_weighted_f1_score_0',   
            'LRPGNNExplainer': BASE_S3 + '29b7f5ee991e4a3e8b553b49a1c3c05a/artifacts/model_best_val_weighted_f1_score_0',
            'GraphGradCAMPPExplainer': BASE_S3 + '29b7f5ee991e4a3e8b553b49a1c3c05a/artifacts/model_best_val_weighted_f1_score_0',
            'GraphGradCAMExplainer': BASE_S3 + '29b7f5ee991e4a3e8b553b49a1c3c05a/artifacts/model_best_val_weighted_f1_score_0'
        },
        'superpx_graph_model': {
            'GNNExplainer': BASE_S3 + 'b61bdbcec5084c2fa4aef79d0515dfc1/artifacts/model_best_val_weighted_f1_score_0',   
            'LRPGNNExplainer': BASE_S3 + 'b61bdbcec5084c2fa4aef79d0515dfc1/artifacts/model_best_val_weighted_f1_score_0',
            'GraphGradCAMPPExplainer': BASE_S3 + 'b61bdbcec5084c2fa4aef79d0515dfc1/artifacts/model_best_val_weighted_f1_score_0',
            'GraphGradCAMExplainer': BASE_S3 + 'b61bdbcec5084c2fa4aef79d0515dfc1/artifacts/model_best_val_weighted_f1_score_0'
        }
    },
    '2_class_scenario': {
        'cell_graph_model': {
            'GNNExplainer': BASE_S3 + 'ac170b8c6da247cea4076b988f3f3218/artifacts/model_best_val_weighted_f1_score_0',   
            'LRPGNNExplainer': BASE_S3 + 'ac170b8c6da247cea4076b988f3f3218/artifacts/model_best_val_weighted_f1_score_0',
            'GraphGradCAMPPExplainer': BASE_S3 + 'ac170b8c6da247cea4076b988f3f3218/artifacts/model_best_val_weighted_f1_score_0',
            'GraphGradCAMExplainer': BASE_S3 + 'ac170b8c6da247cea4076b988f3f3218/artifacts/model_best_val_weighted_f1_score_0'
        },
        'superpx_graph_model': {
            'GNNExplainer': BASE_S3 + '72293c1321a54df18b81672936c79517/artifacts/model_best_val_weighted_f1_score_0',   
            'LRPGNNExplainer': BASE_S3 + '72293c1321a54df18b81672936c79517/artifacts/model_best_val_weighted_f1_score_0',
            'GraphGradCAMPPExplainer': BASE_S3 + '72293c1321a54df18b81672936c79517/artifacts/model_best_val_weighted_f1_score_0',
            'GraphGradCAMExplainer': BASE_S3 + '72293c1321a54df18b81672936c79517/artifacts/model_best_val_weighted_f1_score_0'
        }
    }
}


INTERPRETABILITY_MODEL_TYPE_TO_LOAD_FN = {
    'pruning_explainer.graph_pruning_explainer': 'tentative_model_loading',
    'LRPGNNExplainer': 'plain_model_loading',
    'GraphGradCAMPPExplainer': 'plain_model_loading',
    'GraphGradCAMExplainer': 'plain_model_loading'
}

