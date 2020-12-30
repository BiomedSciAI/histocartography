import logging
from abc import abstractmethod
from typing import Any, List

import numpy as np
import sklearn.metrics
import torch
from histocartography.preprocessing.utils import fast_histogram


class SegmentationMetric:
    """Base class for segmentation metrics"""

    def __init__(self, **kwargs):
        """Constructor of Metric"""

    @abstractmethod
    def _compute_metric(
        self, ground_truth: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Actual metric computation

        Args:
            ground_truth (torch.Tensor): Ground truth tensor. Shape: (B x H x W)
            prediction (torch.Tensor): Prediction tensor. Shape: (B x H x W)

            Returns:
               torch.Tensor: Computed metric. Shape: (1 or B)
        """

    def __call__(
        self, prediction: torch.Tensor, ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """From either a batched, unbatched or batched with additional empty dimension calculate the metric accordingly

        Args:
            ground_truth (torch.Tensor): Ground truth tensor. Shape: (H x W, B x H x W, or B x 1 x H x W)
            prediction (torch.Tensor): Prediction tensor. Shape: (same shape as ground truth)

        Returns:
            torch.Tensor: Computed metric. Shape: (1 or B)
        """
        if isinstance(ground_truth, np.ndarray):
            ground_truth = torch.Tensor(ground_truth)
        if isinstance(prediction, np.ndarray):
            prediction = torch.Tensor(prediction)
        assert ground_truth.shape == prediction.shape
        unbatched = False
        if len(ground_truth.shape) == 4:  # For shape BATCH x 1 x H x W
            assert ground_truth.shape[1] == 1
            prediction = prediction.squeeze(1)
            ground_truth = ground_truth.squeeze(1)
        if len(prediction.shape) == 2:  # For shape H x W
            unbatched = True
            prediction = prediction.unsqueeze(0)
            ground_truth = ground_truth.unsqueeze(0)
        # Now we have shape BATCH x H x W
        metric = self._compute_metric(ground_truth=ground_truth, prediction=prediction)
        if unbatched:
            return metric[0]
        return metric

    @staticmethod
    def is_better(value: Any, comparison: Any) -> bool:
        raise NotImplementedError


class IoU(SegmentationMetric):
    """Compute the class IoU"""

    def __init__(self, nr_classes: int, background_label: int, **kwargs) -> None:
        """Create a IoU calculator for a certain number of classes

        Args:
            nr_classes (int, optional): Number of classes to use
        """
        self.nr_classes = nr_classes
        self.smooth = 1e-12
        self.background_label = background_label
        super().__init__(**kwargs)

    def _compute_metric(
        self,
        ground_truth: torch.Tensor,
        prediction: torch.Tensor,
        nan: float = float("nan"),
    ) -> torch.Tensor:
        """Computes the intersection over union per class

        Args:
            ground_truth (torch.Tensor): Ground truth tensor
            prediction (torch.Tensor): Prediction tensor
            nan (float, optional): Value to use for non-existant class. Defaults to float('nan').

        Returns:
            torch.Tensor: Computed IoU
        """
        class_iou = torch.empty((ground_truth.shape[0], self.nr_classes))
        for class_label in range(self.nr_classes):
            class_ground_truth = ground_truth == class_label
            if class_label == self.background_label:
                class_iou[:, class_label] = nan
                continue
            if not class_ground_truth.any():
                class_iou[:, class_label] = nan
                continue
            class_prediction = prediction == class_label
            class_intersection = (class_ground_truth & class_prediction).sum(
                axis=(1, 2)
            )
            class_union = (class_ground_truth | class_prediction).sum(axis=(1, 2))
            class_iou[:, class_label] = class_intersection / (class_union + self.smooth)
        return class_iou

    @staticmethod
    def is_better(value: Any, comparison: Any) -> bool:
        """Higher is better"""
        return value >= comparison


class MeanIoU(IoU):
    """Mean class IoU"""

    def _compute_metric(
        self, ground_truth: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Computes the average iou over the existing classes in the ground truth
           Same as sklearn.metric.jaccard_score with average='macro', but faster
        Args:
            ground_truth (torch.Tensor): Ground truth tensor
            prediction (torch.Tensor): Prediction tensor

        Returns:
            torch.Tensor: The computed mean IoU per class
        """
        class_iou = super()._compute_metric(ground_truth, prediction)
        mask = torch.isnan(class_iou)
        class_iou[mask] = 0
        batch_mean_iou = torch.sum(class_iou, axis=1) / torch.sum(~mask, axis=1)
        return batch_mean_iou.mean()


class fIoU(IoU):
    """Inverse Log Frequency Weighted IoU as defined in the paper:
    HistoSegNet: Semantic Segmentation of Histological Tissue Typein Whole Slide Images
    Code at: https://github.com/lyndonchan/hsn_v1/
    """

    def _compute_metric(
        self, ground_truth: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Computes the metric according to the implementation:
           https://github.com/lyndonchan/hsn_v1/blob/4356e68fc2a94260ab06e2ceb71a7787cba8178c/hsn_v1/hsn_v1.py#L184

        Args:
            ground_truth (torch.Tensor): Ground truth tensor
            prediction (torch.Tensor): Prediction tensor

        Returns:
            torch.Tensor: Computed fIoU
        """
        class_counts = torch.empty((ground_truth.shape[0], self.nr_classes))
        for class_label in range(self.nr_classes):
            class_ground_truth = ground_truth == class_label
            class_counts[:, class_label] = class_ground_truth.sum(axis=(1, 2))
        class_iou = super()._compute_metric(ground_truth, prediction, nan=1.0)

        log_class_counts = torch.max(torch.zeros_like(class_counts), class_counts.log())
        y = log_class_counts.sum() / log_class_counts
        y[log_class_counts == 0] = 0
        class_weights = y / y.sum()
        batch_fiou = (class_weights * class_iou).sum(axis=1)
        return batch_fiou.mean()


class F1Score(SegmentationMetric):
    """Compute the class F1 score"""

    def __init__(self, nr_classes: int = 5, **kwargs) -> None:
        """Create a F1 calculator for a certain number of classes

        Args:
            nr_classes (int, optional): Number of classes to use. Defaults to 5.
        """
        self.nr_classes = nr_classes
        self.smooth = 1e-12
        super().__init__(**kwargs)

    def _compute_metric(
        self,
        ground_truth: torch.Tensor,
        prediction: torch.Tensor,
        nan: float = float("nan"),
    ) -> torch.Tensor:
        """Computes the f1 score per class

        Args:
            ground_truth (torch.Tensor): Ground truth tensor
            prediction (torch.Tensor): Prediction tensor
            nan (float, optional): Value to use for non-existant class. Defaults to float('nan').

        Returns:
            torch.Tensor: Computed F1 scores
        """
        class_f1 = torch.empty((ground_truth.shape[0], self.nr_classes))
        for class_label in range(self.nr_classes):
            class_ground_truth = ground_truth == class_label
            if not class_ground_truth.any():
                class_f1[:, class_label] = nan
                continue
            class_prediction = prediction == class_label
            true_positives = (class_ground_truth & class_prediction).sum(axis=(1, 2))
            false_positives = (
                torch.logical_not(class_ground_truth) & class_prediction
            ).sum(axis=(1, 2))
            false_negatives = (
                class_ground_truth & torch.logical_not(class_prediction)
            ).sum(axis=(1, 2))
            precision = true_positives / (
                true_positives + false_positives + self.smooth
            )
            recall = true_positives / (true_positives + false_negatives + self.smooth)
            class_f1[:, class_label] = (2.0 * precision * recall) / (
                precision + recall + self.smooth
            )

        return class_f1

    @staticmethod
    def is_better(value: Any, comparison: Any) -> bool:
        """Higher is better"""
        return value >= comparison


class MeanF1Score(F1Score):
    """Mean class F1 score"""

    def _compute_metric(
        self, ground_truth: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Computes the average f1 score over the existing classes in the ground truth
           Same as sklearn.metric.f1_score with average='macro', but faster
        Args:
            ground_truth (torch.Tensor): Ground truth tensor
            prediction (torch.Tensor): Prediction tensor

        Returns:
            torch.Tensor: The computed mean f1 score per class
        """
        class_f1 = super()._compute_metric(ground_truth, prediction)
        mask = torch.isnan(class_f1)
        class_f1[mask] = 0
        batch_f1 = torch.sum(class_f1, axis=1) / torch.sum(~mask, axis=1)
        return batch_f1.mean()


class ClassificationMetric:
    """Base class for classification metrics"""

    def __init__(self, *args, **kwargs):
        """Constructor of Metric"""
        logging.info(f"Unmatched keyword arguments for metric: {kwargs}")

    @abstractmethod
    def _compute_metric(
        self, logits: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> Any:
        """Actual metric computation

        Args:
            ground_truth (torch.Tensor): Ground truth tensor. Shape: (B x H x W)
            prediction (torch.Tensor): Prediction tensor. Shape: (B x H x W)

            Returns:
               Any: Metric value
        """

    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self._compute_metric(logits, labels, **kwargs)

    @staticmethod
    def is_better(value: Any, comparison: Any) -> bool:
        """Higher is better"""
        return value >= comparison


class MultiLabelClassificationMetric(ClassificationMetric):
    def __init__(self, **kwargs) -> None:
        logging.info(f"Unmatched keyword arguments for metric: {kwargs}")

    @abstractmethod
    def _compare(self, predictions, labels, **kwargs) -> float:
        pass

    def _compute_metric(
        self, graph_logits: torch.Tensor, graph_labels: torch.Tensor, **kwargs
    ) -> float:
        """Compute the loss of the logit and targets

        Args:
            graph_logits (torch.Tensor): Logits for the graph with the shape: B x nr_classes
            graph_targets (torch.Tensor): Targets one-hot encoded with the shape: B x nr_classes

        Returns:
            float: Graph loss
        """
        predictions = torch.sigmoid(graph_logits)
        return self._compare(predictions, graph_labels, **kwargs)


class MultiLabelAccuracy(MultiLabelClassificationMetric):
    def _compare(self, predictions, labels, **kwargs):
        y_pred = np.ravel(predictions.numpy()) > 0.5
        y_true = np.ravel(labels.numpy())
        return sklearn.metrics.accuracy_score(y_pred=y_pred, y_true=y_true)


class MultiLabelBalancedAccuracy(MultiLabelClassificationMetric):
    def _compare(self, predictions, labels, **kwargs):
        y_pred = np.ravel(predictions.numpy()) > 0.5
        y_true = np.ravel(labels.numpy())
        return sklearn.metrics.balanced_accuracy_score(y_pred=y_pred, y_true=y_true)


class MultiLabelF1Score(MultiLabelClassificationMetric):
    def _compare(self, predictions, labels, **kwargs):
        y_pred = np.ravel(predictions.numpy()) > 0.5
        y_true = np.ravel(labels.numpy())
        return sklearn.metrics.f1_score(y_pred=y_pred, y_true=y_true, average="macro")


class MultiLabelAUCROC(MultiLabelClassificationMetric):
    def _compare(self, predictions, labels, **kwargs):
        y_pred = np.ravel(predictions.numpy())
        y_true = np.ravel(labels.numpy())
        return sklearn.metrics.roc_auc_score(
            y_score=y_pred, y_true=y_true, average="macro"
        )


class MultiLabelBenignAccuracy(MultiLabelBalancedAccuracy):
    def _compare(self, predictions, labels, **kwargs) -> float:
        return super()._compare(predictions[:, 0], labels[:, 0], **kwargs)


class MultiLabelGleason3Accuracy(MultiLabelBalancedAccuracy):
    def _compare(self, predictions, labels, **kwargs) -> float:
        return super()._compare(predictions[:, 1], labels[:, 1], **kwargs)


class MultiLabelGleason4Accuracy(MultiLabelBalancedAccuracy):
    def _compare(self, predictions, labels, **kwargs) -> float:
        return super()._compare(predictions[:, 2], labels[:, 2], **kwargs)


class MultiLabelGleason5Accuracy(MultiLabelBalancedAccuracy):
    def _compare(self, predictions, labels, **kwargs) -> float:
        return super()._compare(predictions[:, 3], labels[:, 3], **kwargs)


class NodeClassificationMetric(ClassificationMetric):
    def __init__(self, background_label, nr_classes, *args, **kwargs) -> None:
        self.background_label = background_label
        self.nr_classes = nr_classes
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _compare(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> float:
        pass

    def _compute_metric(
        self, node_logits: torch.Tensor, node_labels: torch.Tensor, **kwargs
    ) -> float:
        predictions = torch.softmax(node_logits, dim=1)
        return self._compare(predictions, node_labels, **kwargs)


class NodeClassificationAccuracy(NodeClassificationMetric):
    def _compare(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        node_associations: List[int],
        **kwargs,
    ) -> float:
        accuracies = np.empty(len(node_associations))
        start = 0
        for i, node_association in enumerate(node_associations):
            y_pred = np.argmax(
                predictions[start : start + node_association, ...].numpy(), axis=1
            )
            y_true = labels[start : start + node_association].numpy()
            mask = y_true != self.background_label
            accuracies[i] = sklearn.metrics.accuracy_score(
                y_pred=y_pred[mask], y_true=y_true[mask]
            )
            start += node_association
        return np.mean(accuracies[accuracies == accuracies])

    
class NodeClassificationBalancedAccuracy(NodeClassificationMetric):
    def _compare(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        node_associations: List[int],
        **kwargs,
    ) -> float:
        accuracies = np.empty(len(node_associations))
        start = 0
        for i, node_association in enumerate(node_associations):
            y_pred = np.argmax(
                predictions[start : start + node_association, ...].numpy(), axis=1
            )
            y_true = labels[start : start + node_association].numpy()
            mask = y_true != self.background_label
            accuracies[i] = sklearn.metrics.balanced_accuracy_score(
                y_pred=y_pred[mask], y_true=y_true[mask]
            )
            start += node_association
        return np.mean(accuracies[accuracies == accuracies])


class NodeClassificationF1Score(NodeClassificationMetric):
    def _compare(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        node_associations: List[int],
        **kwargs,
    ) -> float:
        accuracies = np.empty(len(node_associations))
        start = 0
        for i, node_association in enumerate(node_associations):
            y_pred = np.argmax(
                predictions[start : start + node_association, ...].numpy(), axis=1
            )
            y_true = labels[start : start + node_association].numpy()
            mask = y_true != self.background_label
            accuracies[i] = sklearn.metrics.f1_score(
                y_pred=y_pred[mask], y_true=y_true[mask], average="weighted"
            )
            start += node_association
        return np.mean(accuracies[accuracies == accuracies])


class NodeClassificationBalancedAccuracy(NodeClassificationMetric):
    def _compare(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        node_associations: List[int],
        **kwargs,
    ) -> float:
        accuracies = np.empty(len(node_associations))
        start = 0
        for i, node_association in enumerate(node_associations):
            y_pred = np.argmax(
                predictions[start : start + node_association, ...].numpy(), axis=1
            )
            y_true = labels[start : start + node_association].numpy()
            mask = y_true != self.background_label
            accuracies[i] = sklearn.metrics.balanced_accuracy_score(
                y_pred=y_pred[mask], y_true=y_true[mask]
            )
            start += node_association
        return np.mean(accuracies[accuracies == accuracies])


class NodeClassificationF1Score(NodeClassificationMetric):
    def _compare(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        node_associations: List[int],
        **kwargs,
    ) -> float:
        accuracies = np.empty(len(node_associations))
        start = 0
        for i, node_association in enumerate(node_associations):
            y_pred = np.argmax(
                predictions[start : start + node_association, ...].numpy(), axis=1
            )
            y_true = labels[start : start + node_association].numpy()
            mask = y_true != self.background_label
            accuracies[i] = sklearn.metrics.f1_score(
                y_pred=y_pred[mask], y_true=y_true[mask], average="weighted"
            )
            start += node_association
        return np.mean(accuracies[accuracies == accuracies])


def sum_up_gleason(annotation, n_class=4, thres=0):
    # read the mask and count the grades
    grade_count = fast_histogram(annotation.flatten(), n_class)
    grade_count = grade_count / grade_count.sum()
    grade_count[grade_count < thres] = 0

    # get the max and second max scores and write them to file
    idx = np.argsort(grade_count)
    primary_score = idx[-1]
    secondary_score = idx[-2]

    if np.sum(grade_count == 0) == n_class - 1:
        secondary_score = primary_score
    if secondary_score == 0:
        secondary_score = primary_score
    if primary_score == 0:
        primary_score = secondary_score

    # Fix scores
    if primary_score + secondary_score == 0:
        return 0
    else:
        return primary_score + secondary_score - 1


# Legacy compatibility (remove in the future)
GraphClassificationAccuracy = MultiLabelAccuracy
GraphClassificationBalancedAccuracy = MultiLabelBalancedAccuracy
GraphClassificationF1Score = MultiLabelF1Score
GraphClassificationAUCROC = MultiLabelAUCROC
GraphClassificationBenignAccuracy = MultiLabelBenignAccuracy
GraphClassificationGleason3Accuracy = MultiLabelGleason3Accuracy
GraphClassificationGleason4Accuracy = MultiLabelGleason4Accuracy
GraphClassificationGleason5Accuracy = MultiLabelGleason5Accuracy
