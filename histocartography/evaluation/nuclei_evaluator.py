import torch
import numpy as np
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

    def __call__(self, node_importance, troi_label, nuclei_labels):  # node importance ?
        """
        Go from this format: 
        0: {'nuclei_label': tumorous, 'importance': 0.23, 'troi_label': invasive}, etc...
        to 
        'tumorous': {'benign': [0.23, 0.12, 0.34], 'pathological_benign': [0.23, 0.12, 0.34],}
        """
        nuclei_info = {}
        v = defaultdict(list)
        for _, value in sorted(nuclei_info.items()):
            v[value['nuclei_label']].append([value['troi_label'], value['importance']])
        
        sorted_nuclei_info = defaultdict(list)
        for nuclei_type, value in v.items():
            for entry in value:
                sorted_nuclei_info[nuclei_type][entry[0]].append(value[1])

        out = {}
        for nuclei_type, value in sorted_nuclei_info.items():
            out[nuclei_type] = {}
            for tumour_type, importances in value.items():
                out[nuclei_type][tumour_type] = {
                    'mean': torch.mean(importances),
                    'std': torch.std(importances),
                    'all': importances
                }
        
        return out


class ComputeKLDivSeparability(BaseEvaluator):
    """
    Compute the mean/std per nuclei type per tumor type 
    """

    def __init__(self, cuda=False):
        super(ComputeKLDivSeparability, self).__init__(cuda)

    def __call__(self, nuclei_info):  
        """
        """
        for nuclei_type, value in nuclei_info.items():
            for tumour_type_1, info_1 in value.items():
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
                        nuclei_info[nuclei_type][tumour_type_1]['kl'] = {tumour_type_2: kl_div}
        return nuclei_info


class ComputeAggKLDivSeparability(BaseEvaluator):
    """
    Compute the agg mean/std per nuclei type per tumor type 
    """

    def __init__(self, cuda=False):
        super(ComputeAggKLDivSeparability, self).__init__(cuda)
        self.compute_kl = ComputeKLDivSeparability(cuda)

    def __call__(self, nuclei_info):  
        """
        """
        nuclei_info = self.compute_kl(nuclei_info)
        for nuclei_type, value in nuclei_info.items():
            for tumour_type, info in value.items():
                as_list = [val for key, val in info['kl'].items()]
                info['kl']['agg'] = torch.mean(torch.FloatTensor(as_list))
        return nuclei_info
