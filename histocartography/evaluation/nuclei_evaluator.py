import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from histocartography.evaluation.base_evaluator import BaseEvaluator


class BaseNucleiEvaluator(BaseEvaluator):
    """
    Compute the relevance between the node importance and the nuclei type.
    """

    def __init__(self, cuda=False):
        super(BaseNucleiEvaluator, self).__init__(cuda)

    def __call__(self, node_importance, troi_label, nuclei_labels):  # node importance ?
        return None
