import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, f1_score
from collections import defaultdict
from histocartography.evaluation.base_evaluator import BaseEvaluator


class BaseNucleiEvaluator(BaseEvaluator):
    """
    Compute the relevance between the node importance and the nuclei type.
    """

    def __init__(self, cuda=False):
        super(BaseNucleiEvaluator, self).__init__(cuda)

    def __call__(self, node_importance, troi_label, nuclei_labels):  # node importance ?
        return None


class ComputeMeanStdPerNukPerTumor(BaseEvaluator):
    """
    Compute the mean/std per nuclei type per tumor type 
    """

    def __init__(self, cuda=False):
        super(ComputeMeanStdPerNukPerTumor, self).__init__(cuda)
        self.eval_kl_separability = ComputeKLDivSeparability()
        self.eval_agg_kl_separability = ComputeAggKLDivSeparability()

    def __call__(self, nuclei_info, troi_labels): 
        """
        Go from this format: 
        0: {'nuclei_label': tumorous, 'importance': 0.23, 'troi_label': invasive}, etc...
        to 
        'tumorous': {'benign': [0.23, 0.12, 0.34], 'pathological_benign': [0.23, 0.12, 0.34],}
        """
        v = defaultdict(list)
        for _, value in sorted(nuclei_info.items()):
            v[value['nuclei_label']].append([value['troi_label'], value['nuclei_importance']])

        sorted_nuclei_info = {
            0: defaultdict(list),
            1: defaultdict(list),
            2: defaultdict(list),
            3: defaultdict(list),
            4: defaultdict(list),
            5: defaultdict(list),
        }
        for nuclei_type, value in v.items():
            for entry in value:
                sorted_nuclei_info[nuclei_type][entry[0]].append(entry[1])

        nuclei_stats_per_nuk_per_tumor = {}
        for nuclei_type, value in sorted_nuclei_info.items():
            nuclei_stats_per_nuk_per_tumor[nuclei_type] = {}
            for tumour_type, importances in value.items():
                nuclei_stats_per_nuk_per_tumor[nuclei_type][tumour_type] = {
                    'mean': torch.mean(torch.FloatTensor(importances)),
                    'std': torch.std(torch.FloatTensor(importances)),
                    'all': torch.FloatTensor(importances)
                }
        nuclei_stats_per_nuk_per_tumor = self.eval_kl_separability(nuclei_stats_per_nuk_per_tumor, troi_labels)
        nuclei_stats_per_nuk_per_tumor = self.eval_agg_kl_separability(nuclei_stats_per_nuk_per_tumor, troi_labels)
        nuclei_stats_per_nuk_per_tumor = apply_recursive(to_serializable, nuclei_stats_per_nuk_per_tumor)
        
        return nuclei_stats_per_nuk_per_tumor


class ComputeTumorSimilarity(BaseEvaluator):
    """
    Compute the similarity between tumor types based on a given concept and 
    its importance (for more details, refer to the implementation).
    """

    def __init__(self, cuda=False):
        super(ComputeTumorSimilarity, self).__init__(cuda)

    def __call__(self, nuclei_info, concept_name='nuclei_label'): 
        """

        Param
        :param nuclei_info: (dict) expected format is:
            {
                0: {'nuclei_label': tumorous, 'importance': 0.23, 'troi_label': invasive},
                1: {'nuclei_label': tumorous, 'importance': 0.42, 'troi_label': invasive},
                2: {...}
            }
        :param concept_name: (str) key to extract from the nuclei info param
        """

        v = defaultdict(list)
        for _, value in sorted(nuclei_info.items()):
            v[value['troi_label']].append([value[concept_name], value['nuclei_importance']])

        num_classes = 3  # len(list(v.keys()))
        nuclei_stats_per_tumor = torch.zeros(num_classes, num_classes)

        # apply one hot encoding if categorical concept variable 
        if concept_name == 'nuclei_label':  # only categorical concept so far
            for tumor_type, data in v.items():
                concept = to_one_hot(np.array(data[0]))

        for tumor_type_row, row_data in v.items():
            for tumor_type_col, col_data in v.items():
                nuclei_stats_per_tumor[tumor_type_row, tumor_type_col] = compute_point_cloud_similarity(np.array(row_data), np.array(col_data))

        # convert to serializable type (ie list of floats)
        nuclei_stats_per_tumor = list(nuclei_stats_per_tumor.cpu().detach().numpy())
        
        return nuclei_stats_per_tumor


def to_one_hot(data):
    """
    Concert (categorical) data to one hot encoding.
    """
    data = data.astype(int)
    one_hot = np.zeros((data.size, int(data.max()+1)))
    one_hot[np.arange(data.size), data] = 1
    return one_hot


def compute_point_cloud_similarity(data1, data2):
    """
    Compute the point cloud similarity between 2 sets of points.
    The similarity is based on the L2 bidirectional Chamfer loss.

    Note: for categorical variables (eg nuclei type), use a one-hot encoding
          of the category. 

    Param:
    :param data1: (ndarray) [n_points_1, n_dims]
    :param data1: (ndarray) [n_points_2, n_dims]

    :return chamfer_dist: (float) the chamfer distance between the 2 points clouds
    """

    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(data1)
    min_y_to_x = x_nn.kneighbors(data2)[0]
    y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(data2)
    min_x_to_y = y_nn.kneighbors(data1)[0]
    chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)

    return chamfer_dist

def to_serializable(val):
    if isinstance(val, torch.Tensor):
        if torch.numel(val) == 1:
            val = float(val.item())
        else:
            val = [float(x) for x in (list(val.cpu().detach().numpy()))]
    return val


def apply_recursive(func, obj):
    if isinstance(obj, dict):  # if dict, apply to each key
        return {k: apply_recursive(func, v) for k, v in obj.items()}
    elif isinstance(obj, list):  # if list, apply to each element
        return [apply_recursive(func, elem) for elem in obj]
    else:
        return func(obj)


class ComputeKLDivSeparability(BaseEvaluator):
    """
    Compute the mean/std per nuclei type per tumor type 
    """

    def __init__(self, cuda=False):
        super(ComputeKLDivSeparability, self).__init__(cuda)

    def __call__(self, nuclei_stats, nuclei_labels):  
        """
        """
        for nuclei_type, value in nuclei_stats.items():
            for tumour_type_1, info_1 in value.items():
                info_1['kl'] = {}
                info_1['kl']['all'] = []
                for tumour_type_2, info_2 in value.items():
                    if tumour_type_1 != tumour_type_2:
                        p = torch.distributions.Normal(
                            info_1['mean'],
                            info_1['std']
                        )
                        q = torch.distributions.Normal(
                            info_2['mean'],
                            info_2['std']
                        )
                        kl_div = torch.distributions.kl_divergence(p, q).mean()
                        info_1['kl']['all'].append(kl_div)
        return nuclei_stats


class ComputeAggKLDivSeparability(BaseEvaluator):
    """
    Compute the agg mean/std per nuclei type per tumor type 
    """

    def __init__(self, cuda=False):
        super(ComputeAggKLDivSeparability, self).__init__(cuda)
        self.compute_kl = ComputeKLDivSeparability(cuda)

    def __call__(self, nuclei_stats, nuclei_labels):  
        """
        """
        for nuclei_type, value in nuclei_stats.items():
            for tumour_type, info in value.items():
                info['kl']['agg'] = torch.mean(torch.FloatTensor(info['kl']['all']))
        return nuclei_stats
