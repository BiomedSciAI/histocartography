
# define all the available modules for interpretability 
AVAILABLE_EXPLAINABILITY_METHODS = {
    'pruning_explainer.graph_pruning_explainer': 'SingleInstanceExplainer',
    'lrp_explainer.lrp_gnn_explainer': 'LRPGNNExplainer',
    'attention_based_explainer.attention_gnn_explainer': 'AttentionGNNExplainer',
}
