import torch
from sklearn.metrics import confusion_matrix


class BaseEvaluator:
    """
    Base interface class for evaluation metrics.
    """

    def __init__(self, cuda=False):
        """
        Base Evaluator constructor.

        Args:
            cuda: (bool) if cuda is available
        """
        super(BaseEvaluator, self).__init__()
        self.cuda = cuda

    def __call__(self, logits, labels):
        """
        Evaluate
        Args:
            objects: (list) each element in the list is a dict with:
                - centroid
                - label
                - visual descriptor
            image_size: (list) weight and height of the image
        """
        raise NotImplementedError("Implementation in sub classes.")


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
        super(AccuracyEvaluator, self).__init__(cuda)

    def __call__(self, logits, labels):
        _, indices = torch.max(logits, dim=1)
        conf_matrix = confusion_matrix(labels.cpu().numpy(), indices.cpu().numpy())
        return conf_matrix
