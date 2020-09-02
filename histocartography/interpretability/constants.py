import networkx as nx


# define all the available modules for interpretability 
AVAILABLE_EXPLAINABILITY_METHODS = {
    'pruning_explainer.graph_pruning_explainer': 'GraphPruningExplainer',
    'lrp_explainer.lrp_gnn_explainer': 'LRPGNNExplainer',
    'attention_based_explainer.attention_gnn_explainer': 'AttentionGNNExplainer',
    'saliency_explainer.graph_gradcam_explainer': 'GraphGradCAMExplainer'
}


MODEL_TYPE_TO_GNN_LAYER_NAME = {
    'superpx_graph_model': 'superpx_gnn',
    'cell_graph_model': 'cell_graph_gnn'
}


BASE_S3 = 's3://mlflow/'


# Assuming the class split is the 5-class scenario one 
MODEL_TO_MLFLOW_ID = {
    '5_class_scenario': {
        'cell_graph_model': {
            'attention_based_explainer.attention_gnn_explainer': BASE_S3 + '70a419364cda43548a6e3733ad0ef0b5/artifacts/model_best_val_weighted_f1_score_0',
            'pruning_explainer.graph_pruning_explainer': BASE_S3 + '7ad2792ad69940e0a54dc554af3c4716/artifacts/model_best_val_weighted_f1_score_0',   # '55ebbd5a765e4a1d80818112212e1875/artifacts/model_best_val_loss_0',
            'lrp_explainer.lrp_gnn_explainer': BASE_S3 + '7ad2792ad69940e0a54dc554af3c4716/artifacts/model_best_val_weighted_f1_score_0',
            'saliency_explainer.graph_gradcam_explainer': BASE_S3 + '7ad2792ad69940e0a54dc554af3c4716/artifacts/model_best_val_weighted_f1_score_0'
        },
        'multi_level_graph_model': {
            'lrp_explainer.lrp_gnn_explainer': BASE_S3 + '5ced95f2b389478cb3e57aaa1c77fc94/artifacts/model_best_val_loss_0',
            'saliency_explainer.graph_gradcam_explainer': BASE_S3 + '',
            'attention_based_explainer.attention_gnn_explainer': BASE_S3 + '',
            'pruning_explainer.graph_pruning_explainer': BASE_S3 + ''
        },
        'superpx_graph_model': {
            'lrp_explainer.lrp_gnn_explainer': BASE_S3 + '96343b43e4284334910c8901258262d4/artifacts/model_best_val_weighted_f1_score_0',
            'saliency_explainer.graph_gradcam_explainer': BASE_S3 + '96343b43e4284334910c8901258262d4/artifacts/model_best_val_weighted_f1_score_0',
            'attention_based_explainer.attention_gnn_explainer': BASE_S3 + '18fd64661fc84067bf2598d67dcad5f6/artifacts/model_best_val_weighted_f1_score_0',
            'pruning_explainer.graph_pruning_explainer': BASE_S3 + '96343b43e4284334910c8901258262d4/artifacts/model_best_val_weighted_f1_score_0'
        }
    },
    '3_class_scenario': {
        'cell_graph_model': {
            'attention_based_explainer.attention_gnn_explainer': BASE_S3 + 'bf7d631ec3ba4a8db10a566c85e094e0/artifacts/model_best_val_weighted_f1_score_0',
            'pruning_explainer.graph_pruning_explainer': BASE_S3 + '29b7f5ee991e4a3e8b553b49a1c3c05a/artifacts/model_best_val_weighted_f1_score_0',   
            'lrp_explainer.lrp_gnn_explainer': BASE_S3 + '29b7f5ee991e4a3e8b553b49a1c3c05a/artifacts/model_best_val_weighted_f1_score_0',
            'saliency_explainer.graph_gradcam_explainer': BASE_S3 + '29b7f5ee991e4a3e8b553b49a1c3c05a/artifacts/model_best_val_weighted_f1_score_0'
        },
        'multi_level_graph_model': {
            'attention_based_explainer.attention_gnn_explainer': BASE_S3 + '',
            'pruning_explainer.graph_pruning_explainer': BASE_S3 + '',   
            'lrp_explainer.lrp_gnn_explainer': BASE_S3 + '',
            'saliency_explainer.graph_gradcam_explainer': BASE_S3 + ''
        },
        'superpx_graph_model': {
            'attention_based_explainer.attention_gnn_explainer': BASE_S3 + '22b18c0a22b4461784c1fae07f4581e2/artifacts/model_best_val_weighted_f1_score_0',
            'pruning_explainer.graph_pruning_explainer': BASE_S3 + 'b61bdbcec5084c2fa4aef79d0515dfc1/artifacts/model_best_val_weighted_f1_score_0',   
            'lrp_explainer.lrp_gnn_explainer': BASE_S3 + 'b61bdbcec5084c2fa4aef79d0515dfc1/artifacts/model_best_val_weighted_f1_score_0',
            'saliency_explainer.graph_gradcam_explainer': BASE_S3 + 'b61bdbcec5084c2fa4aef79d0515dfc1/artifacts/model_best_val_weighted_f1_score_0'
        }
    },
    '2_class_scenario': {
        'cell_graph_model': {
            'attention_based_explainer.attention_gnn_explainer': BASE_S3 + '57b7ce4984774f7f85e2cdc24794051a/artifacts/model_best_val_weighted_f1_score_0',
            'pruning_explainer.graph_pruning_explainer': BASE_S3 + 'ac170b8c6da247cea4076b988f3f3218/artifacts/model_best_val_weighted_f1_score_0',   
            'lrp_explainer.lrp_gnn_explainer': BASE_S3 + 'ac170b8c6da247cea4076b988f3f3218/artifacts/model_best_val_weighted_f1_score_0',
            'saliency_explainer.graph_gradcam_explainer': BASE_S3 + 'ac170b8c6da247cea4076b988f3f3218/artifacts/model_best_val_weighted_f1_score_0'
        },
        'multi_level_graph_model': {
            'attention_based_explainer.attention_gnn_explainer': BASE_S3 + '',
            'pruning_explainer.graph_pruning_explainer': BASE_S3 + '',   
            'lrp_explainer.lrp_gnn_explainer': BASE_S3 + '',
            'saliency_explainer.graph_gradcam_explainer': BASE_S3 + ''
        },
        'superpx_graph_model': {
            'attention_based_explainer.attention_gnn_explainer': BASE_S3 + '493af367ab98406f957b1891a5ebdef3/artifacts/model_best_val_weighted_f1_score_0',
            'pruning_explainer.graph_pruning_explainer': BASE_S3 + '72293c1321a54df18b81672936c79517/artifacts/model_best_val_weighted_f1_score_0',   
            'lrp_explainer.lrp_gnn_explainer': BASE_S3 + '72293c1321a54df18b81672936c79517/artifacts/model_best_val_weighted_f1_score_0',
            'saliency_explainer.graph_gradcam_explainer': BASE_S3 + '72293c1321a54df18b81672936c79517/artifacts/model_best_val_weighted_f1_score_0'
        }
    }
}


INTERPRETABILITY_MODEL_TYPE_TO_LOAD_FN = {
    'attention_based_explainer.attention_gnn_explainer': 'plain_model_loading',
    'pruning_explainer.graph_pruning_explainer': 'tentative_model_loading',
    'lrp_explainer.lrp_gnn_explainer': 'plain_model_loading',
    'saliency_explainer.graph_gradcam_explainer': 'plain_model_loading'
}


EXPLANATION_TYPE_SAVE_SUBDIR = {
    'attention_based_explainer.attention_gnn_explainer': 'GAT',
    'pruning_explainer.graph_pruning_explainer': 'GNNExplainer',
    'lrp_explainer.lrp_gnn_explainer': 'GraphLRP',
    'saliency_explainer.graph_gradcam_explainer': 'GraphGradCAMExplainer'
}


# define KG encding the class inter-dependencies for 5-class problem 
FIVE_CLASS_NAMES = ['benign', 'pathologicalbenign', 'atypical', 'dcis', 'malignant']
FIVE_CLASS_DEPENDENCY_GRAPH = nx.Graph()
for idx, label in enumerate(FIVE_CLASS_NAMES):
    FIVE_CLASS_DEPENDENCY_GRAPH.add_node(idx, attr={'name': label})    

FIVE_CLASS_DEPENDENCY_GRAPH.add_edge(0, 1)
FIVE_CLASS_DEPENDENCY_GRAPH.add_edge(1, 2)
FIVE_CLASS_DEPENDENCY_GRAPH.add_edge(1, 3)
FIVE_CLASS_DEPENDENCY_GRAPH.add_edge(2, 3)
FIVE_CLASS_DEPENDENCY_GRAPH.add_edge(3, 4)


# define KG encding the class inter-dependencies for 5-class problem 
SEVEN_CLASS_NAMES = ['benign', 'pathologicalbenign', 'udh', 'adh', 'fea', 'dcis', 'malignant']
SEVEN_CLASS_DEPENDENCY_GRAPH = nx.Graph()
for idx, label in enumerate(SEVEN_CLASS_NAMES):
    SEVEN_CLASS_DEPENDENCY_GRAPH.add_node(label, attr={'index': idx})    

SEVEN_CLASS_DEPENDENCY_GRAPH.add_edge('benign', 'pathologicalbenign')
SEVEN_CLASS_DEPENDENCY_GRAPH.add_edge('pathologicalbenign', 'udh')
SEVEN_CLASS_DEPENDENCY_GRAPH.add_edge('pathologicalbenign', 'adh')
SEVEN_CLASS_DEPENDENCY_GRAPH.add_edge('pathologicalbenign', 'fea')
SEVEN_CLASS_DEPENDENCY_GRAPH.add_edge('fea', 'dcis')
SEVEN_CLASS_DEPENDENCY_GRAPH.add_edge('udh', 'adh')
SEVEN_CLASS_DEPENDENCY_GRAPH.add_edge('adh', 'dcis')
SEVEN_CLASS_DEPENDENCY_GRAPH.add_edge('dcis', 'malignant')


KEEP_PERCENTAGE_OF_NODE_IMPORTANCE = [1, 0.5]
