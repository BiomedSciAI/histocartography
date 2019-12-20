import dgl

NORMALIZATION_FACTORS = {
    'cell_graph': {
    },
    'superpx_graph': {
        'mean': [],
        'std': []
    }
}


COLLATE_FN = {
    'DGLGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x
}

# TUMOR_TYPE_TO_LABEL = {
#     'Benign': 0,
#     'DCIS': 1,
#     'Malignant': 2
# }

TUMOR_TYPE_TO_LABEL = {
    'ADH': 0,
    'FEA': 1,
    'Malignant': 2
}
