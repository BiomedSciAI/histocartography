import dgl
import torch
import os 


NORMALIZATION_FACTORS = {
    'features_hc_': {
        'cell_graph': {
            'mean':
            torch.tensor([
                9.4237e+01, 5.7058e+01, 4.0923e+02, 5.8869e-01, 4.2775e+00,
                2.8502e+03, 5.5121e+01, 2.1028e+01, 5.6037e+02, 7.2949e-01,
                3.5301e+02, 2.6564e+01, 1.6340e+01, 7.3179e+01, 9.3862e-01,
                8.7872e+01
            ]),
            'std':
            torch.tensor([
                1.6193e+01, 2.3741e+01, 2.3299e+02, 4.6444e-01, 1.4010e-01,
                1.4889e+03, 3.5600e+01, 7.5556e+00, 7.5240e+02, 1.5308e-01,
                1.7888e+02, 1.1263e+01, 5.0055e+00, 2.1249e+01, 4.7270e-02,
                5.5668e+01
            ])
        },
        'superpx_graph': {
            'mean':
            torch.tensor([
                2.1981e+04, 3.7003e+04, 7.0996e-01, 1.1810e+02, 9.0075e-01,
                5.9065e-01, 2.4262e+04, 1.8072e+02, 1.0477e+02, 6.1989e-03,
                8.3046e+02, 7.8040e-01, 3.0056e-05, 5.2342e-03, 4.6998e-02,
                9.2689e-02, 1.2871e-01, 1.6947e-01, 1.9739e-01, 3.5948e-01,
                1.9232e+02, 4.4891e+01, 1.9881e+02, -6.6931e-01, 9.0838e+01,
                1.8622e-03, 6.1457e-02, 1.3770e-01, 1.8241e-01, 1.8166e-01,
                1.5252e-01, 1.1852e-01, 1.6387e-01, 1.5208e+02, 4.8407e+01,
                1.5258e+02, -1.7227e-02, 9.9288e+01, 3.4702e-06, 4.6681e-04,
                2.3783e-02, 1.1691e-01, 2.1831e-01, 2.4339e-01, 1.9305e-01,
                2.0409e-01, 1.7921e+02, 3.6415e+01, 1.8072e+02, -1.6544e-01,
                9.9342e+01, 4.1816e+00, 2.7873e+02, 1.1188e+01, 1.5125e-01,
                4.7216e-02, 8.1969e-03
            ]),
            'std':
            torch.tensor([
                1.2505e+05, 2.2614e+05, 1.7029e-01, 1.1818e+02, 1.0489e+00,
                1.3246e-01, 1.5343e+05, 2.4484e+02, 1.3306e+02, 9.1175e-01,
                2.4338e+03, 1.1259e-01, 2.2788e-04, 8.6511e-03, 4.6669e-02,
                6.3628e-02, 6.0786e-02, 6.1151e-02, 6.5038e-02, 2.0853e-01,
                2.4549e+01, 9.1727e+00, 2.9905e+01, 8.8591e-01, 1.8297e+01,
                4.8199e-03, 6.2920e-02, 8.4730e-02, 7.9891e-02, 6.9107e-02,
                6.2152e-02, 6.3559e-02, 1.9336e-01, 3.2826e+01, 8.4669e+00,
                4.0336e+01, 7.7069e-01, 1.3678e+01, 3.0714e-05, 1.7742e-03,
                3.2247e-02, 8.7976e-02, 1.0244e-01, 9.1583e-02, 8.5106e-02,
                2.0822e-01, 2.4877e+01, 6.6230e+00, 2.9747e+01, 7.4435e-01,
                1.4869e+01, 4.6195e-01, 1.0825e+02, 2.7396e+00, 1.0675e-01,
                7.5780e-02, 3.3562e-02
            ])
        }
    },
    'vae': {
        'cell_graph': {
            'mean': torch.tensor([-0.7471,  0.4554,  0.6270,  0.0626,  0.1339, -1.9104, -0.5668,  0.4720,
         1.6469,  0.9699, -3.8639,  4.3818, -1.4651, -0.7107,  0.8660, -0.2592,
        10.7798, -2.2976, -4.3065, -1.4636, -6.1647, 16.4574,  2.4411, -1.6574,
         1.1240,  0.5186,  5.6994, -1.7227,  2.1603, -0.0667,  0.8356, -2.6960]),
            'std': torch.tensor([2.3207, 2.4710, 2.2681, 2.0871, 2.5357, 2.1358, 3.6012, 2.2947, 1.9796,
        2.2289, 3.3571, 3.1551, 3.2585, 1.7618, 2.3605, 2.6159, 3.7951, 1.8653,
        3.3419, 2.0905, 3.3388, 4.4166, 1.5699, 5.9051, 2.2171, 2.2057, 4.9633,
        3.3894, 6.2628, 2.3896, 2.1222, 8.5666]),
        },
        'superpx_graph': {
            'mean': torch.ones(32) * 0.5,
            'std': torch.ones(32) * 0.5,
        }
    }
}


COLLATE_FN = {
    'DGLGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x,
    'PngImageFile': lambda x: x,
    'str': lambda x: x
}


NODE_FEATURE_TYPE_TO_DIRNAME = {
    'features_cnn_resnet34_mask_True_': os.path.join('nuclei_features', 'features_cnn_resnet34_mask_True_'), 
    'features_cnn_resnet34_mask_False_': os.path.join('nuclei_features', 'features_cnn_resnet34_mask_False_'), 
    'features_cnn_resnet101_mask_False_': os.path.join('nuclei_features', 'features_cnn_resnet101_mask_False_'), 
    'features_cnn_resnet50_mask_False_': os.path.join('nuclei_features', 'features_cnn_resnet50_mask_False_'), 
    'features_cnn_vgg16_mask_False_': os.path.join('nuclei_features', 'features_cnn_vgg16_mask_False_'), 
    'features_cnn_vgg19_mask_False_': os.path.join('nuclei_features', 'features_cnn_vgg19_mask_False_'), 
    'features_hc_': os.path.join('nuclei_features', 'features_hc_'), 
    'features_cnn_resnet101_mask_True_': os.path.join('nuclei_features', 'features_cnn_resnet101_mask_True_'), 
    'features_cnn_resnet50_mask_True_': os.path.join('nuclei_features', 'features_cnn_resnet50_mask_True_'), 
    'features_cnn_vgg16_mask_True_': os.path.join('nuclei_features', 'features_cnn_vgg16_mask_True_'), 
    'features_cnn_vgg19_mask_True_': os.path.join('nuclei_features', 'features_cnn_vgg19_mask_True_'), 
    'nuclei_vae_features': os.path.join('nuclei_features', 'nuclei_vae_features'),
    'centroid': os.path.join('nuclei_detected', 'centroids')
}

NODE_FEATURE_TYPE_TO_H5 = {
    'features_cnn_resnet34_mask_True_': 'embeddings', 
    'features_cnn_resnet34_mask_False_': 'embeddings', 
    'features_cnn_resnet101_mask_False_': 'embeddings', 
    'features_cnn_resnet50_mask_False_': 'embeddings', 
    'features_cnn_vgg16_mask_False_': 'embeddings', 
    'features_cnn_vgg19_mask_False_': 'embeddings', 
    'features_hc_': 'instance_features', 
    'features_cnn_resnet101_mask_True_': 'embeddings', 
    'features_cnn_resnet50_mask_True_': 'embeddings', 
    'features_cnn_vgg16_mask_True_': 'embeddings', 
    'features_cnn_vgg19_mask_True_': 'embeddings', 
    'nuclei_vae_features': 'vae_embeddings',
    'centroid': 'instance_centroid_location'
}


def get_tumor_type_to_label(class_split):

    # get the classes 
    grouped_classes = class_split.split('VS')

    # build mapping 
    tumor_type_to_label = {c: group_idx for group_idx, group in enumerate(grouped_classes) for c in group.split('+')}
    return tumor_type_to_label


def get_number_of_classes(class_split):
    return len(class_split.split('VS'))


def get_label_to_tumor_type(class_split):

    # get the classes 
    grouped_classes = class_split.split('VS')

    # build mapping 
    tumor_type_to_label = {group_idx: str(c) for group_idx, group in enumerate(grouped_classes) for c in group.split('+')}
    return tumor_type_to_label


ALL_DATASET_NAMES = ['adh', 'benign', 'dcis', 'fea', 'malignant', 'pathologicalbenign', 'udh']


def get_dataset_white_list(class_split):
    white_classes = class_split.replace('VS', '+').split('+')
    return white_classes


# def get_dataset_black_list(class_split):
#     white_classes = class_split.replace('VS', '+').split('+')
#     black_classes = [item for item in ALL_DATASET_NAMES if item not in white_classes]
#     return black_classes


NUM_CLASSES_TO_MODEL_URL = {
    2: '9eab3cda4e324254b5044fe4c0b90368/artifacts/model_best_val_weighted_f1_score_3',
    3: '0550391249d941588ed547235ca84046/artifacts/model_best_val_weighted_f1_score_3',
    5: 'd504d8ba7e7848098c7562a72e98e7bd/artifacts/model_best_val_weighted_f1_score_3'
}
