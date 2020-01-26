import dgl
import torch

NORMALIZATION_FACTORS = {
    'cell_graph': {'mean': torch.tensor([9.3564e+01, 6.4326e+01, 4.2089e+02, 6.0968e-01, 4.3258e+00, 2.0927e+03,
                                         4.0686e+01, 1.7329e+01, 3.8951e+02, 7.1562e-01, 2.5981e+02, 2.2306e+01,
                                         1.3914e+01, 6.1413e+01, 9.4021e-01, 8.7411e+01]),
                   'std': torch.tensor([1.5778e+01, 2.2866e+01, 2.4106e+02, 4.6837e-01, 1.2226e-01, 1.2184e+03,
                                        2.6063e+01, 5.9187e+00, 2.9949e+02, 1.5391e-01, 1.4410e+02, 8.3460e+00,
                                        4.2261e+00, 1.8914e+01, 4.6122e-02, 5.4640e+01])},
    'superpx_graph': {
        'mean': torch.FloatTensor([0.5] * 57),
        'std': torch.FloatTensor([0.5] * 57)
    }
}


COLLATE_FN = {
    'DGLGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x,
    'PngImageFile': lambda x: x,
    'str': lambda x: x
}


# tumor type to 4. Currently UDH, ADH, FEA and DCIS are grouped under the same label
TUMOR_TYPE_TO_LABEL = {
    'pathological_benign': 0,
    'benign': 1,
    'udh': 2,
    'adh': 2,
    'fea': 2,
    'dcis': 3,
    'malignant': 4
}


# List of classes to discard for training
DATASET_BLACKLIST = ['dcis', 'adh', 'fea', 'udh', 'malignant']

DATASET_TO_TUMOR_TYPE = {
    '0_benign': 'benign',
    '1_pathological_benign': 'pathological_benign',
    '2_udh': 'udh',
    '3_adh': 'adh',
    '4_fea': 'fea',
    '5_dcis': 'dcis',
    '6_malignant': 'malignant'
}
