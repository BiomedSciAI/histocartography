import dgl
import torch


NORMALIZATION_FACTORS = {
    'cell_graph': {'mean': torch.tensor([9.4237e+01, 5.7058e+01, 4.0923e+02, 5.8869e-01, 4.2775e+00, 2.8502e+03,
                                         5.5121e+01, 2.1028e+01, 5.6037e+02, 7.2949e-01, 3.5301e+02, 2.6564e+01,
                                         1.6340e+01, 7.3179e+01, 9.3862e-01, 8.7872e+01]),
                   'std': torch.tensor([1.6193e+01, 2.3741e+01, 2.3299e+02, 4.6444e-01, 1.4010e-01, 1.4889e+03,
                                        3.5600e+01, 7.5556e+00, 7.5240e+02, 1.5308e-01, 1.7888e+02, 1.1263e+01,
                                        5.0055e+00, 2.1249e+01, 4.7270e-02, 5.5668e+01])},
    'superpx_graph': {'mean': torch.tensor([2.1981e+04,  3.7003e+04,  7.0996e-01,  1.1810e+02,  9.0075e-01,
                                            5.9065e-01,  2.4262e+04,  1.8072e+02,  1.0477e+02,  6.1989e-03,
                                            8.3046e+02,  7.8040e-01,  3.0056e-05,  5.2342e-03,  4.6998e-02,
                                            9.2689e-02,  1.2871e-01,  1.6947e-01,  1.9739e-01,  3.5948e-01,
                                            1.9232e+02,  4.4891e+01,  1.9881e+02, -6.6931e-01,  9.0838e+01,
                                            1.8622e-03,  6.1457e-02,  1.3770e-01,  1.8241e-01,  1.8166e-01,
                                            1.5252e-01,  1.1852e-01,  1.6387e-01,  1.5208e+02,  4.8407e+01,
                                            1.5258e+02, -1.7227e-02,  9.9288e+01,  3.4702e-06,  4.6681e-04,
                                            2.3783e-02,  1.1691e-01,  2.1831e-01,  2.4339e-01,  1.9305e-01,
                                            2.0409e-01,  1.7921e+02,  3.6415e+01,  1.8072e+02, -1.6544e-01,
                                            9.9342e+01,  4.1816e+00,  2.7873e+02,  1.1188e+01,  1.5125e-01,
                                            4.7216e-02,  8.1969e-03]),
                      'std': torch.tensor([1.2505e+05, 2.2614e+05, 1.7029e-01, 1.1818e+02, 1.0489e+00, 1.3246e-01,
                                           1.5343e+05, 2.4484e+02, 1.3306e+02, 9.1175e-01, 2.4338e+03, 1.1259e-01,
                                           2.2788e-04, 8.6511e-03, 4.6669e-02, 6.3628e-02, 6.0786e-02, 6.1151e-02,
                                           6.5038e-02, 2.0853e-01, 2.4549e+01, 9.1727e+00, 2.9905e+01, 8.8591e-01,
                                           1.8297e+01, 4.8199e-03, 6.2920e-02, 8.4730e-02, 7.9891e-02, 6.9107e-02,
                                           6.2152e-02, 6.3559e-02, 1.9336e-01, 3.2826e+01, 8.4669e+00, 4.0336e+01,
                                           7.7069e-01, 1.3678e+01, 3.0714e-05, 1.7742e-03, 3.2247e-02, 8.7976e-02,
                                           1.0244e-01, 9.1583e-02, 8.5106e-02, 2.0822e-01, 2.4877e+01, 6.6230e+00,
                                           2.9747e+01, 7.4435e-01, 1.4869e+01, 4.6195e-01, 1.0825e+02, 2.7396e+00,
                                           1.0675e-01, 7.5780e-02, 3.3562e-02])}
    }


COLLATE_FN = {
    'DGLGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x,
    'PngImageFile': lambda x: x,
    'str': lambda x: x
}


def get_tumor_type_to_label(num_classes=5):

    if num_classes == 2:
        tumor_type_to_label = {
            'benign': 0,
            'pathologicalbenign': 0,
            'udh': 0,
            'dcis': 1,
            'malignant': 1
        }

    elif num_classes == 3:
        tumor_type_to_label = {
            'benign': 0,
            'pathologicalbenign': 0,
            'udh': 0,
            'dcis': 1,
            'malignant': 1,
            'adh': 2,
            'fea': 2
        }

    elif num_classes == 5:
        tumor_type_to_label = {
            'benign': 0,
            'pathologicalbenign': 1,
            'udh': 1,
            'dcis': 2,
            'malignant': 3,
            'adh': 4,
            'fea': 4
        }

    else:
        raise ValueError('Number of classes can 2, 3 or 5.')

    return tumor_type_to_label


def get_label_to_tumor_type(num_classes=5):

    if num_classes == 2:
        label_to_tumor_type = {
            '0': 'N',
            '1': 'I',
        }

    elif num_classes == 3:
        label_to_tumor_type = {
            '0': 'N',
            '1': 'I',
            '2': 'ATY'
        }

    elif num_classes == 5:
        label_to_tumor_type = {
            '0': 'N',
            '1': 'B',
            '2': 'DCIS',
            '3': 'I',
            '4': 'ATY'
        }

    else:
        raise ValueError('Number of classes can 2, 3 or 5.')

    return label_to_tumor_type


def get_dataset_black_list(num_classes=5):
    if num_classes == 2:
        dataset_blacklist = ['fea', 'adh']

    elif num_classes == 3:
        dataset_blacklist = []

    elif num_classes == 5:
        dataset_blacklist = []

    else:
        raise ValueError('Number of classes can 2, 3 or 5.')

    return dataset_blacklist


NUM_CLASSES_TO_MODEL_URL = {
    2: 'a2f387ebe35c4bb98989470af00c12a3/artifacts/model_best_val_loss_3',
    3: 'd504d8ba7e7848098c7562a72e98e7bd/artifacts/model_best_val_weighted_f1_score_3',
    5: 'd504d8ba7e7848098c7562a72e98e7bd/artifacts/model_best_val_weighted_f1_score_3'
}