import logging
from abc import abstractmethod
from typing import Any, List, Union

import numpy as np
import sklearn.metrics
import torch
from histocartography.preprocessing.utils import fast_histogram


class Metric:
    def __init__(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def is_better(value: Any, comparison: Any) -> bool:
        raise NotImplementedError

    @property
    def logs_model(self):
        return True

    @property
    def is_per_class(self):
        return False


class SegmentationMetric(Metric):
    """Base class for segmentation metrics"""

    def __init__(self, background_label, **kwargs):
        """Constructor of Metric"""
        self.background_label = background_label
        super().__init__(**kwargs)

    @abstractmethod
    def _compute_metric(
        self, ground_truth: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Actual metric computation

        Args:
            ground_truth (torch.Tensor): Ground truth tensor. Shape: (B x H x W)
            prediction (torch.Tensor): Prediction tensor. Shape: (B x H x W)

            Returns:
               torch.Tensor: Computed metric. Shape: (B)
        """

    def __call__(
        self,
        prediction: Union[torch.Tensor, np.ndarray],
        ground_truth: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ) -> torch.Tensor:
        """From either a batched, unbatched calculate the metric accordingly and take the average over the samples

        Args:
            ground_truth (Union[torch.Tensor, np.ndarray]): Ground truth tensor. Shape: (H x W, B x H x W)
            prediction (Union[torch.Tensor, np.ndarray]): Prediction tensor. Shape: (same shape as ground truth)

        Returns:
            torch.Tensor: Computed metric
        """
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.detach().cpu().numpy()
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        assert ground_truth.shape == prediction.shape

        if len(prediction.shape) == 2:
            prediction = prediction[np.newaxis, :, :]
            ground_truth = ground_truth[np.newaxis, :, :]
        # Now we have shape BATCH x H x W

        # Discard background class
        prediction_copy = prediction.copy()
        prediction_copy[ground_truth == self.background_label] = self.background_label

        metric = self._compute_metric(
            ground_truth=ground_truth, prediction=prediction_copy
        )
        return np.nanmean(metric, axis=0)


class IoU(SegmentationMetric):
    """Compute the class IoU"""

    def __init__(self, nr_classes: int, background_label: int, **kwargs) -> None:
        """Create a IoU calculator for a certain number of classes

        Args:
            nr_classes (int, optional): Number of classes to use
        """
        self.nr_classes = nr_classes
        self.smooth = 1e-12
        super().__init__(background_label=background_label, **kwargs)

    def _compute_sample_metric(
        self,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        nan: float = float("nan"),
    ):
        assert (
            ground_truth.shape == prediction.shape
        ), f"Prediction and ground truth must have same shape, but is {ground_truth.shape}, {prediction.shape}"
        assert (
            len(ground_truth.shape) == 2
        ), f"Expected 2D tensor but got {ground_truth.shape}"
        class_iou = np.empty(self.nr_classes)
        for class_label in range(self.nr_classes):
            class_ground_truth = ground_truth == class_label
            if not class_ground_truth.any():
                class_iou[class_label] = nan
                continue
            class_prediction = prediction == class_label
            class_intersection = (class_ground_truth & class_prediction).sum(
                axis=(0, 1)
            )
            class_union = (class_ground_truth | class_prediction).sum(axis=(0, 1))
            class_iou[class_label] = class_intersection / (class_union + self.smooth)
        return class_iou

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
        class_ious = list()
        for i in range(ground_truth.shape[0]):
            class_ious.append(
                self._compute_sample_metric(
                    prediction=prediction[i], ground_truth=ground_truth[i], nan=nan
                )
            )
        return np.stack(class_ious)

    @staticmethod
    def is_better(value: Any, comparison: Any) -> bool:
        """Higher is better"""
        return value >= comparison

    @property
    def is_per_class(self):
        return True


class MeanIoU(IoU):
    """Mean class IoU"""

    def __call__(
        self,
        prediction: Union[torch.Tensor, np.ndarray],
        ground_truth: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        return np.nanmean(super().__call__(prediction, ground_truth))


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

    def _compute_sample_metric(
        self,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        nan: float = float("nan"),
    ):
        class_f1 = np.empty(self.nr_classes)
        for class_label in range(self.nr_classes):
            class_ground_truth = ground_truth == class_label
            if not class_ground_truth.any():
                class_f1[class_label] = nan
                continue
            class_prediction = prediction == class_label
            true_positives = (class_ground_truth & class_prediction).sum(axis=(0, 1))
            false_positives = (
                np.logical_not(class_ground_truth) & class_prediction
            ).sum(axis=(0, 1))
            false_negatives = (
                class_ground_truth & np.logical_not(class_prediction)
            ).sum(axis=(0, 1))
            precision = true_positives / (
                true_positives + false_positives + self.smooth
            )
            recall = true_positives / (true_positives + false_negatives + self.smooth)
            class_f1[class_label] = (2.0 * precision * recall) / (
                precision + recall + self.smooth
            )
        return class_f1

    def _compute_metric(
        self,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
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
        class_f1s = list()
        for i in range(ground_truth.shape[0]):
            class_f1s.append(
                self._compute_sample_metric(
                    prediction=prediction[i], ground_truth=ground_truth[i], nan=nan
                )
            )
        return np.stack(class_f1s)

    @staticmethod
    def is_better(value: Any, comparison: Any) -> bool:
        """Higher is better"""
        return value >= comparison

    @property
    def is_per_class(self):
        return True


class MeanF1Score(F1Score):
    """Mean class F1 score"""

    def __call__(
        self,
        prediction: Union[torch.Tensor, np.ndarray],
        ground_truth: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        return np.nanmean(super().__call__(prediction, ground_truth))


class ClassificationMetric(Metric):
    """Base class for classification metrics"""

    def __init__(self, *args, **kwargs):
        """Constructor of Metric"""
        logging.info(f"Unmatched keyword arguments for metric: {kwargs}")
        super().__init__(*args, **kwargs)

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
        return value >= comparison


class MultiLabelClassificationMetric(ClassificationMetric):
    def __init__(self, nr_classes: int, **kwargs) -> None:
        self.nr_classes = nr_classes
        super().__init__(**kwargs)

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
        assert (
            predictions.shape == graph_labels.shape
        ), f"Must be same shape, but got: {predictions.shape}, {graph_labels.shape}"
        return self._compare(predictions, graph_labels, **kwargs)


class MultiLabelSklearnMetric(MultiLabelClassificationMetric):
    def __init__(self, f, threshold, nr_classes: int, **kwargs) -> None:
        super().__init__(nr_classes, **kwargs)
        self.f = f
        self.threshold = threshold

    def _compare(self, predictions, labels, **kwargs):
        assert (
            len(predictions.shape) == 2
        ), f"Must be 2D tensor, but got: {predictions.shape}"
        class_metric = np.empty(self.nr_classes)
        for i in range(self.nr_classes):
            y_pred = predictions.numpy()[:, i]
            y_true = labels[:, i].numpy()
            if self.threshold:
                class_metric[i] = self.f(y_pred=y_pred > 0.5, y_true=y_true)
            else:
                class_metric[i] = self.f(y_score=y_pred, y_true=y_true)
        return class_metric

    @property
    def is_per_class(self):
        return True


class MultiLabelAccuracy(MultiLabelSklearnMetric):
    def __init__(self, nr_classes: int, **kwargs) -> None:
        super().__init__(
            f=sklearn.metrics.accuracy_score,
            threshold=True,
            nr_classes=nr_classes,
            **kwargs,
        )


class MultiLabelBalancedAccuracy(MultiLabelSklearnMetric):
    def __init__(self, nr_classes: int, **kwargs) -> None:
        super().__init__(
            f=sklearn.metrics.balanced_accuracy_score,
            threshold=True,
            nr_classes=nr_classes,
            **kwargs,
        )


class MultiLabelF1Score(MultiLabelSklearnMetric):
    def __init__(self, nr_classes: int, **kwargs) -> None:
        super().__init__(
            f=sklearn.metrics.f1_score, threshold=True, nr_classes=nr_classes, **kwargs
        )


class NodeClassificationMetric(ClassificationMetric):
    def __init__(self, background_label, nr_classes, *args, **kwargs) -> None:
        self.background_label = background_label
        self.nr_classes = nr_classes
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _compare(self, prediction, ground_truth):
        pass

    def _compute_metric(
        self,
        node_logits: torch.Tensor,
        node_labels: torch.Tensor,
        node_associations: List[int],
    ) -> float:
        predictions = torch.softmax(node_logits, dim=1)
        metrics = np.empty(len(node_associations))
        start = 0
        for i, node_association in enumerate(node_associations):
            metrics[i] = self._compare(
                prediction=predictions[start : start + node_association, ...].numpy(),
                ground_truth=node_labels[start : start + node_association].numpy(),
            )
            start += node_association
        return np.nanmean(metrics)


class NodeClassificationsSklearnMetric(NodeClassificationMetric):
    def __init__(self, f, background_label, nr_classes, average=None, **kwargs) -> None:
        super().__init__(background_label, nr_classes, **kwargs)
        self.f = f
        self.average = average

    def _compare(self, prediction, ground_truth):
        y_pred = np.argmax(prediction, axis=1)
        mask = ground_truth != self.background_label
        if self.average is not None:
            return self.f(
                y_pred=y_pred[mask], y_true=ground_truth[mask], average=self.average
            )
        else:
            return self.f(y_pred=y_pred[mask], y_true=ground_truth[mask])


class NodeClassificationAccuracy(NodeClassificationsSklearnMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(sklearn.metrics.accuracy_score, *args, **kwargs)


class NodeClassificationBalancedAccuracy(NodeClassificationsSklearnMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(sklearn.metrics.balanced_accuracy_score, *args, **kwargs)


class NodeClassificationF1Score(NodeClassificationsSklearnMetric):
    def __init__(self, **kwargs) -> None:
        super().__init__(sklearn.metrics.f1_score, average="weighted", **kwargs)


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


class GleasonScoreMetric(Metric):
    def __init__(self, f, nr_classes: int, background_label: int, **kwargs) -> None:
        """Create a IoU calculator for a certain number of classes

        Args:
            nr_classes (int, optional): Number of classes to use
        """
        self.nr_classes = nr_classes
        self.background_label = background_label
        self.f = f
        self.kwargs = kwargs
        super().__init__()

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        tissue_mask=None,
        **kwargs,
    ) -> Any:
        assert prediction.shape == ground_truth.shape
        assert len(prediction.shape) == 3
        assert tissue_mask is None or len(tissue_mask) == prediction.shape[0]

        gleason_grade_ground_truth = list()
        gleason_grade_prediction = list()
        for i, (logits, labels) in enumerate(zip(prediction, ground_truth)):
            if tissue_mask is not None:
                logits[~tissue_mask[i]] = self.background_label

            gleason_grade_ground_truth.append(
                sum_up_gleason(labels, n_class=self.nr_classes)
            )
            gleason_grade_prediction.append(
                sum_up_gleason(logits, n_class=self.nr_classes, thres=0.25)
            )
        return self.f(
            gleason_grade_ground_truth,
            gleason_grade_prediction,
            **self.kwargs,
        )

    @staticmethod
    def is_better(value: Any, comparison: Any) -> bool:
        return value >= comparison


class GleasonScoreKappa(GleasonScoreMetric):
    def __init__(self, nr_classes: int, background_label: int) -> None:
        super().__init__(f=sklearn.metrics.cohen_kappa_score, nr_classes=nr_classes, background_label=background_label, weights="quadratic")


class GleasonScoreF1(GleasonScoreMetric):
    def __init__(self, nr_classes: int, background_label: int, **kwargs) -> None:
        super().__init__(f=sklearn.metrics.f1_score, nr_classes=nr_classes, background_label=background_label, average="weighted")
