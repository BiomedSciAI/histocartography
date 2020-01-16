import dgl


NORMALIZATION_FACTORS = {
    'cell_graph': {},
    'superpx_graph': {}
}


COLLATE_FN = {
    'DGLGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x,
    'PngImageFile': lambda x: x
}


# tumor type to 4. Currently UDH, ADH, FEA and DCIS are grouped under the same label
TUMOR_TYPE_TO_LABEL = {
    'pathologicalbenign': 1,
    'benign': 0,
    'udh': 2,
    'adh': 2,
    'fea': 2,
    'dcis': 2,
    'malignant': 3
}


# List of classes to discard for training
DATASET_BLACKLIST = ['udh']
