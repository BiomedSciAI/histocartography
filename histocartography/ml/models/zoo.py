# The models follow the naming convention:
# <dataset_name>_<model_type>_<number_of_classes>_classes_<gnn_type>.pt
# e.g., in bracs_tggnn_7_classes_pna:
# -btrained on bracs dataset
# - with tggnn (Tissue Graph GNN) model
# - and 7 classes
# - with PNA GNN layers

MODEL_NAME_TO_URL = {
    # CG-GNN
    'bracs_cggnn_3_classes_gin.pt': 'https://ibm.box.com/shared/static/pozkx0ngqjxdr34v5tpmckthts8m12u3.pt',
    'bracs_cggnn_5_classes_gin.pt': 'https://ibm.box.com/shared/static/uy2xeovpo001mb1edkwms5ccz1sqhdpz.pt',
    'bracs_cggnn_7_classes_pna.pt': 'https://ibm.box.com/shared/static/i4xixoglstzkif53rc2cm4b568ei5wnf.pt', 
    # TG-GNN
    'bracs_tggnn_3_classes_gin.pt': 'https://ibm.box.com/shared/static/aoogy0516lsp9vaxgw1tr9mdu5nycvvb.pt',
    'bracs_tggnn_7_classes_pna.pt': 'https://ibm.box.com/shared/static/19q7kk2humvc6a8qedzg8rs5bny6qvrf.pt', 
    # HACT
    'bracs_hact_7_classes_pna.pt': 'https://ibm.box.com/shared/static/5v44c33cipdy7c2dhajkrfaywyrh2a5o.pt', 
}

MODEL_NAME_TO_CONFIG = {
    # CG-GNN
    'bracs_cggnn_3_classes_gin.pt': {
        'node_dim': 514,
        'gnn_params': {
            'layer_type': "gin_layer",
            'hidden_dim': 64,
            'output_dim': 64,
            'num_layers': 3,
            'agg_type': "mean",
            'act': "relu",
            'readout_op': "none",
            'readout_type': "mean",
            'batch_norm': False,
            'graph_norm': False,
            'dropout': 0.
        },
        'classification_params': {
            'num_layers': 2,
            'hidden_dim': 128,
        }
    },
    'bracs_cggnn_5_classes_gin.pt': {
        'node_dim': 514,
        'gnn_params': {
            'layer_type': "gin_layer",
            'hidden_dim': 64,
            'output_dim': 64,
            'num_layers': 3,
            'agg_type': "mean",
            'act': "relu",
            'readout_op': "concat",
            'readout_type': "mean",
            'batch_norm': False,
            'graph_norm': False,
            'dropout': 0.
        },
        'classification_params': {
            'num_layers': 2,
            'hidden_dim': 128,
        }
    },
    'bracs_cggnn_7_classes_pna.pt': {
        'node_dim': 514,
        'gnn_params': {
            'layer_type': "pna_layer",
            'output_dim': 64,
            'num_layers': 3,
            'readout_op': "lstm",
            'readout_type': "mean",
            'aggregators': "mean max min std",
            'scalers': "identity amplification attenuation",
            'avg_d': 4,
            'dropout': 0.,
            'graph_norm': True,
            'batch_norm': True,
            'towers': 1,
            'pretrans_layers': 1,
            'posttrans_layers': 1,
            'divide_input': False,
            'residual': False,
        },
        'classification_params': {
            'num_layers': 2,
            'hidden_dim': 128,
        }
    },
    # TG-GNN
    'bracs_tggnn_3_classes_gin.pt': {
        'node_dim': 514,
        'gnn_params': {
            'layer_type': "gin_layer",
            'hidden_dim': 64,
            'output_dim': 64,
            'num_layers': 3,
            'agg_type': "mean",
            'act': "relu",
            'readout_op': "none",
            'readout_type': "mean",
            'batch_norm': False,
            'graph_norm': False,
            'dropout': 0.
        },
        'classification_params': {
            'num_layers': 2,
            'hidden_dim': 128,
        }
    },
    'bracs_tggnn_7_classes_pna.pt': {
        'node_dim': 514,
        'gnn_params': {
            'layer_type': "pna_layer",
            'output_dim': 64,
            'num_layers': 3,
            'readout_op': "lstm",
            'readout_type': "mean",
            'aggregators': "mean max min std",
            'scalers': "identity amplification attenuation",
            'avg_d': 4,
            'dropout': 0.,
            'graph_norm': True,
            'batch_norm': True,
            'towers': 1,
            'pretrans_layers': 1,
            'posttrans_layers': 1,
            'divide_input': False,
            'residual': False,
        },
        'classification_params': {
            'num_layers': 2,
            'hidden_dim': 128,
        }
    },
    # HACT
    'bracs_hact_7_classes_pna.pt': {
        'cg_node_dim': 514,
        'cg_gnn_params': {
            'layer_type': "pna_layer",
            'output_dim': 64,
            'num_layers': 3,
            'readout_op': "lstm",
            'readout_type': "mean",
            'aggregators': "mean max min std",
            'scalers': "identity amplification attenuation",
            'avg_d': 4,
            'dropout': 0.,
            'graph_norm': True,
            'batch_norm': True,
            'towers': 1,
            'pretrans_layers': 1,
            'posttrans_layers': 1,
            'divide_input': False,
            'residual': False,
        },
        'tg_node_dim': 514,
        'tg_gnn_params': {
            'layer_type': "pna_layer",
            'output_dim': 64,
            'num_layers': 3,
            'readout_op': "lstm",
            'readout_type': "mean",
            'aggregators': "mean max min std",
            'scalers': "identity amplification attenuation",
            'avg_d': 4,
            'dropout': 0.,
            'graph_norm': True,
            'batch_norm': True,
            'towers': 1,
            'pretrans_layers': 1,
            'posttrans_layers': 1,
            'divide_input': False,
            'residual': False,
        },
        'classification_params': {
            'num_layers': 2,
            'hidden_dim': 128,
        }
    }
}
