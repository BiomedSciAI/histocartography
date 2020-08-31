import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import networkx as nx 

from histocartography.evaluation.base_evaluator import BaseEvaluator


class AccuracyEvaluator(BaseEvaluator):
    """
    Compute accuracy.
    """

    def __init__(self, cuda=False):
        super(AccuracyEvaluator, self).__init__(cuda)

    def __call__(self, logits, labels):
        _, indices = torch.max(logits, dim=1)
        indices = indices.to(int)
        labels = labels.to(int)
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


class ClusteringQuality(BaseEvaluator):
    """
    Compute weighted F1 score
    """

    def __init__(self, cuda=False):
        super(ClusteringQuality, self).__init__(cuda)

    def __call__(self, embeddings, labels, kg):
        qual = 1.
        return torch.FloatTensor([weighted_f1])


class CrossEntropyLoss(BaseEvaluator):
    """
    Compute weighted F1 score
    """

    def __init__(self, cuda=False):
        super(CrossEntropyLoss, self).__init__(cuda)
        self.ce_loss_eval = torch.nn.CrossEntropyLoss()

    def __call__(self, logits, labels):
        loss = self.ce_loss_eval(logits, labels)
        return loss


class ExpectedClassShiftWithLogits(BaseEvaluator):
    """
    Compute expected class shift 
    """

    def __init__(self, knowledge_graph=None, cuda=False):
        super(ExpectedClassShiftWithLogits, self).__init__(cuda)
        self.knowledge_graph = knowledge_graph
        if knowledge_graph is not None:
            self.shortest_paths = dict(nx.shortest_path_length(knowledge_graph))

    def __call__(self, logits, labels):
        _, indices = torch.max(logits, dim=1)
        indices = indices.to(float)
        labels = labels.to(float)

        if self.knowledge_graph is None:   # we assume a sequence of labels
            class_shift = torch.mean(torch.abs(indices - labels)).cpu()
        else:
            class_shift = torch.mean(torch.FloatTensor(list(map(lambda x: self.shortest_paths.get(x[0].item()).get(x[1].item()), zip(labels, indices)))))
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

