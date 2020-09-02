# define all the available modules for interpretability
AVAILABLE_EXPLAINABILITY_METHODS = {
    'pruning_explainer.graph_pruning_explainer': 'GraphPruningExplainer',
    'lrp_explainer.lrp_gnn_explainer': 'LRPGNNExplainer',
    'attention_based_explainer.attention_gnn_explainer': 'AttentionGNNExplainer',
}


BASE_S3 = 's3://mlflow/'


# Assuming the class split is thr 5-class scenario one
MODEL_TO_MLFLOW_ID = {
    'attention_based_explainer.attention_gnn_explainer': BASE_S3 + '1e28b5968aae41fb9ff1241969f30a83/artifacts/model_best_val_weighted_f1_score_0',
    'pruning_explainer.graph_pruning_explainer': BASE_S3 + '',
    'lrp_explainer.lrp_gnn_explainer': BASE_S3 + '',
    'saliency_explainer.graph_cam': BASE_S3 + ''
}


INTERPRETABILITY_MODEL_TYPE_TO_LOAD_FN = {
    'attention_based_explainer.attention_gnn_explainer': 'plain_model_loading',
    'pruning_explainer.graph_pruning_explainer': 'tentative_model_loading',
    'lrp_explainer.lrp_gnn_explainer': 'plain_model_loading',
    'saliency_explainer.graph_cam': 'plain_model_loading'
}