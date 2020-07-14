import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from histocartography.evaluation.base_evaluator import BaseEvaluator


class AccuracyEvaluator(BaseEvaluator):
    """
    Compute accuracy.
    """

    def __init__(self, cuda=False):
        super(AccuracyEvaluator, self).__init__(cuda)

    def __call__(self, logits, labels):
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        accuracy = correct.item() * 1.0 / len(labels)
        return torch.FloatTensor([accuracy])


class WeightedF1(BaseEvaluator):
    """
    Compute weighted F1 score
    """

    def __init__(self, cuda=False):
        super(WeightedF1, self).__init__(cuda)

    def __call__(self, logits, labels):
        _, indices = torch.max(logits, dim=1)
        indices = indices.to(float)
        weighted_f1 = np.float(f1_score(labels.cpu().numpy(), indices.cpu().numpy(), average='weighted'))
        return torch.FloatTensor([weighted_f1])


class ExpectedClassShiftWithLogits(BaseEvaluator):
    """
    Compute expected class shift 
    """

    def __init__(self, cuda=False):
        super(ExpectedClassShiftWithLogits, self).__init__(cuda)

    def __call__(self, logits, labels):
        _, indices = torch.max(logits, dim=1)
        indices = indices.to(float)
        class_shift = torch.mean(torch.abs(indices - labels)).cpu()
        return class_shift


class ExpectedClassShiftWithHardPred(BaseEvaluator):
    """
    Compute expected class shift 
    """

    def __init__(self, cuda=False):
        super(ExpectedClassShiftWithHardPred, self).__init__(cuda)

    def __call__(self, predictions, labels):
        class_shift = torch.mean(torch.abs(predictions - labels)).cpu()
        return class_shift

