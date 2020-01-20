import dgl
import torch

NORMALIZATION_FACTORS = {
    'cell_graph': {'mean': torch.tensor([9.3564e+01, 6.4326e+01, 4.2089e+02, 6.0968e-01, 4.3258e+00, 2.0927e+03,
                                         4.0686e+01, 1.7329e+01, 3.8951e+02, 7.1562e-01, 2.5981e+02, 2.2306e+01,
                                         1.3914e+01, 6.1413e+01, 9.4021e-01, 8.7411e+01]),
                   'std': torch.tensor([1.5778e+01, 2.2866e+01, 2.4106e+02, 4.6837e-01, 1.2226e-01, 1.2184e+03,
                                        2.6063e+01, 5.9187e+00, 2.9949e+02, 1.5391e-01, 1.4410e+02, 8.3460e+00,
                                        4.2261e+00, 1.8914e+01, 4.6122e-02, 5.4640e+01])},
    'superpx_graph': {}
}


COLLATE_FN = {
    'DGLGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x,
    'PngImageFile': lambda x: x,
    'str': lambda x: x
}


# tumor type to 4. Currently UDH, ADH, FEA and DCIS are grouped under the same label
TUMOR_TYPE_TO_LABEL = {
    'pathologicalbenign': 3,
    'benign': 0,
    'udh': 2,
    'adh': 2,
    'fea': 2,
    'dcis': 2,
    'malignant': 1
}


# List of classes to discard for training
DATASET_BLACKLIST = ['pathological_benign', 'adh', 'fea', 'dcis', 'udh']
