import torch
from sklearn.metrics import confusion_matrix

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
        return accuracy.item()


class ConfusionMatrixEvaluator(BaseEvaluator):
    """
    Compute confusion matrix.
    """

    def __init__(self, cuda=False):
        super(ConfusionMatrixEvaluator, self).__init__(cuda)

    def __call__(self, logits, labels):
        _, indices = torch.max(logits, dim=1)
        conf_matrix = confusion_matrix(labels.cpu().numpy(), indices.cpu().numpy())
        return conf_matrix
