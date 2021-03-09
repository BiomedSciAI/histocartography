from functools import partial
import logging
from abc import abstractmethod
from typing import Any, List, Union
import mlflow

import numpy as np
import sklearn.metrics
import torch
from histocartography.preprocessing.utils import fast_histogram


def fast_confusion_matrix(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor], nr_classes: int):
    """Faster computation of confusion matrix according to https://stackoverflow.com/a/59089379

    Args:
        y_true (Union[np.ndarray, torch.Tensor]): Ground truth (1D)
        y_pred (Union[np.ndarray, torch.Tensor]): Prediction (1D)
        nr_classes (int): Number of classes

    Returns:
        np.ndarray: Confusion matrix of shape nr_classes x nr_classes
    """
    assert y_true.shape == y_pred.shape
    y_true = torch.as_tensor(y_true, dtype=torch.long)
    y_pred = torch.as_tensor(y_pred, dtype=torch.long)
    y = nr_classes * y_true + y_pred
    y = torch.bincount(y)
    if len(y) < nr_classes * nr_classes:
        y = torch.cat((y, torch.zeros(nr_classes * nr_classes - len(y), dtype=torch.long)))
    y = y.reshape(nr_classes, nr_classes)
    return y.numpy()


def inverse_frequency(class_counts: np.ndarray) -> np.ndarray:
    y = np.divide(
        class_counts.sum(axis=1)[:, np.newaxis],
        class_counts,
        out=np.zeros_like(class_counts),
        where=class_counts != 0,
    )
    class_weights = y / y.sum(axis=1)[:, np.newaxis]
    return class_weights


def inverse_log_frequency(class_counts: np.ndarray) -> np.ndarray:
    """Converts class counts into normalized inverse log frequency weights per datapoint

    Args:
        class_counts (np.ndarray): Class counts of shape B x C

    Returns:
        np.ndarray: Class weights of shape B x C
    """
    log_class_counts = np.log(
        class_counts,
        out=np.zeros_like(class_counts),
        where=class_counts != 0,
    )
    return inverse_frequency(log_class_counts)


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
        tissue_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        """From either a batched, unbatched calculate the metric accordingly and take the average over the samples

        Args:
            ground_truth (Union[torch.Tensor, np.ndarray]): Ground truth tensor. Shape: (H x W, B x H x W)
            prediction (Union[torch.Tensor, np.ndarray]): Prediction tensor. Shape: (same shape as ground truth)

        Returns:
            torch.Tensor: Computed metric
        """
        if isinstance(ground_truth, list):
            assert len(ground_truth) == len(prediction)
            prediction_copy = list()
            ground_truth_copy = list()
            for i, (sample_gt, sample_pred) in enumerate(zip(ground_truth, prediction)):
                if isinstance(sample_gt, torch.Tensor):
                    sample_gt = sample_gt.detach().cpu().numpy()
                if isinstance(sample_pred, torch.Tensor):
                    sample_pred = sample_pred.detach().cpu().numpy()
                assert isinstance(
                    sample_gt, np.ndarray
                ), f"Ground truth sample must be np.ndarray but got {type(sample_gt)}"
                assert isinstance(
                    sample_pred, np.ndarray
                ), f"Prediction must be of np.ndarray but got {type(sample_pred)}"
                sample_pred = sample_pred.copy()
                sample_gt = sample_gt.copy()
                if tissue_mask is not None:
                    assert isinstance(tissue_mask, np.ndarray) or isinstance(
                        tissue_mask, list
                    ), f"Tissue mask must be of type np.ndarray or list but got {type(tissue_mask)}"
                    assert isinstance(
                        tissue_mask[i], np.ndarray
                    ), f"Tissue mask entry must be of type np.ndarray, but got {type(tissue_mask[i])}"
                    assert (
                        tissue_mask[i].dtype == bool
                    ), f"Tissue mask must be of dtype bool, but is {tissue_mask[i].dtype}"
                    assert (
                        sample_gt.shape == tissue_mask[i].shape
                    ), f"Shape of sample and mask mismatch: {sample_gt.shape}, {tissue_mask[i].shape}"
                    sample_gt[~tissue_mask[i]] = self.background_label
                sample_pred[sample_gt == self.background_label] = self.background_label
                prediction_copy.append(sample_pred)
                ground_truth_copy.append(sample_gt)
            assert len(prediction) == len(prediction_copy)
            assert len(ground_truth) == len(ground_truth_copy)
            metric = self._compute_metric(
                ground_truth=ground_truth_copy, prediction=prediction_copy
            )
        else:
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
            ground_truth = ground_truth.copy()
            ground_truth[~np.asarray(tissue_mask)] = self.background_label
            prediction_copy = prediction.copy()
            prediction_copy[
                ground_truth == self.background_label
            ] = self.background_label

            metric = self._compute_metric(
                ground_truth=ground_truth, prediction=prediction_copy
            )
        return np.nanmean(metric, axis=0)

    def _get_class_counts(
        self, ground_truth: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        """Computed class counts for each class and datapoint

        Args:
            ground_truth (Union[np.ndarray, List[np.ndarray]]):
                Ground truth tensor of shape B x H x W
                or list of length B of tensors of shape H x W

        Returns:
            np.ndarray: Class weights of shape: B x C
        """
        if isinstance(ground_truth, np.ndarray):
            class_counts = np.empty((ground_truth.shape[0], self.nr_classes))
            for class_label in range(self.nr_classes):
                class_ground_truth = ground_truth == class_label
                class_counts[:, class_label] = class_ground_truth.sum(axis=(1, 2))

        else:  # Iterative version to support jagged tensors
            class_counts = list()
            for gt in ground_truth:
                sample_class_counts = np.empty(self.nr_classes)
                for class_label in range(self.nr_classes):
                    class_ground_truth = gt == class_label
                    sample_class_counts[class_label] = class_ground_truth.sum()
                class_counts.append(sample_class_counts)
            class_counts = np.stack(class_counts)
        return class_counts


class ConfusionMatrixMetric(Metric):
    def __init__(self, nr_classes: int, background_label: int, **kwargs) -> None:
        self.nr_classes = nr_classes
        self.background_label = background_label
        super().__init__(**kwargs)

    def _aggregate(self, confusion_matrix):
        return confusion_matrix

    def __call__(
        self,
        prediction: Union[torch.Tensor, np.ndarray],
        ground_truth: Union[torch.Tensor, np.ndarray],
        tissue_mask=None,
        **kwargs,
    ) -> Any:
        assert len(ground_truth) == len(prediction)

        confusion_matrix = np.zeros((self.nr_classes, self.nr_classes), dtype=np.int64)
        for i, (sample_gt, sample_pred) in enumerate(zip(ground_truth, prediction)):
            if isinstance(sample_gt, torch.Tensor):
                sample_gt = sample_gt.detach().cpu().numpy()
            if isinstance(sample_pred, torch.Tensor):
                sample_pred = sample_pred.detach().cpu().numpy()
            sample_pred = sample_pred.copy()
            sample_gt = sample_gt.copy()
            if tissue_mask is not None:
                sample_gt[~tissue_mask[i]] = self.background_label

            mask = sample_gt != self.background_label
            sample_confusion_matrix = fast_confusion_matrix(
                y_true=sample_gt[mask],
                y_pred=sample_pred[mask],
                nr_classes=self.nr_classes,
            )
            confusion_matrix = confusion_matrix + sample_confusion_matrix
        return self._aggregate(confusion_matrix.T)


class DatasetDice(ConfusionMatrixMetric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.smooth = 1e-12

    def _aggregate(self, confusion_matrix):
        scores = np.empty(self.nr_classes)
        indices = np.arange(self.nr_classes)
        for i in range(self.nr_classes):
            TP = confusion_matrix[i, i]
            index = np.zeros_like(confusion_matrix, dtype=bool)
            index[indices == i, :] = True
            index[i, i] = False
            FP = confusion_matrix[index.astype(bool)].sum()
            index = np.zeros_like(confusion_matrix, dtype=bool)
            index[:, indices == i] = True
            index[i, i] = False
            FN = confusion_matrix[index.astype(bool)].sum()
            recall = TP / (FN + TP + self.smooth)
            precision = TP / (TP + FP + self.smooth)
            scores[i] = 2 * 1 / (1 / (recall + self.smooth) + 1 / (precision + self.smooth) + self.smooth)
        return scores

    @staticmethod
    def is_better(value: Any, comparison: Any) -> bool:
        return value >= comparison

    @property
    def logs_model(self):
        return True

    @property
    def is_per_class(self):
        return True


class DatasetIoU(ConfusionMatrixMetric):
    def _aggregate(self, confusion_matrix):
        scores = np.empty(self.nr_classes)
        indices = np.arange(self.nr_classes)
        for i in range(self.nr_classes):
            TP = confusion_matrix[i, i]
            index = np.zeros_like(confusion_matrix, dtype=bool)
            index[indices == i, :] = True
            index[i, i] = False
            FP = confusion_matrix[index.astype(bool)].sum()
            index = np.zeros_like(confusion_matrix, dtype=bool)
            index[:, indices == i] = True
            index[i, i] = False
            FN = confusion_matrix[index.astype(bool)].sum()
            scores[i] = TP / (TP + FP + FN)
        return scores

    @staticmethod
    def is_better(value: Any, comparison: Any) -> bool:
        return value >= comparison

    @property
    def logs_model(self):
        return False

    @property
    def is_per_class(self):
        return True


class IoU(SegmentationMetric):
    """Compute the class IoU"""

    def __init__(
        self,
        nr_classes: int,
        background_label: int,
        discard_threshold: int = 0,
        **kwargs,
    ) -> None:
        """Create a IoU calculator for a certain number of classes

        Args:
            nr_classes (int, optional): Number of classes to use
        """
        self.nr_classes = nr_classes
        self.discard_threshold = discard_threshold
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
            if class_ground_truth.sum() <= self.discard_threshold:
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
        for pred, gt in zip(prediction, ground_truth):
            class_ious.append(
                self._compute_sample_metric(prediction=pred, ground_truth=gt, nan=nan)
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
        **kwargs,
    ) -> torch.Tensor:
        return np.nanmean(super().__call__(prediction, ground_truth, **kwargs))

    @property
    def is_per_class(self):
        return False


class fIoU(IoU):
    """Inverse Log Frequency Weighted IoU as defined in the paper:
    HistoSegNet: Semantic Segmentation of Histological Tissue Typein Whole Slide Images
    Code at: https://github.com/lyndonchan/hsn_v1/
    """

    def _compute_metric(
        self,
        ground_truth: Union[np.ndarray, List[np.ndarray]],
        prediction: Union[np.ndarray, List[np.ndarray]],
    ) -> float:
        """Computes the metric according to the implementation:
           https://github.com/lyndonchan/hsn_v1/blob/4356e68fc2a94260ab06e2ceb71a7787cba8178c/hsn_v1/hsn_v1.py#L184

        Args:
            ground_truth (Union[np.ndarray, List[np.ndarray]]): Ground truth tensor or list of tensors
            prediction (Union[np.ndarray, List[np.ndarray]]): Prediction tensor or list of tensors

        Returns:
            float: Inverse Log Frequency Weighted IoU
        """
        class_ious = super()._compute_metric(ground_truth, prediction, np.NaN)
        class_counts = self._get_class_counts(ground_truth)
        class_weights = inverse_log_frequency(class_counts)
        return np.nansum(class_weights * class_ious, axis=1)

    @property
    def is_per_class(self):
        return False


class F1Score(SegmentationMetric):
    """Compute the class F1 score"""

    def __init__(
        self, nr_classes: int = 5, discard_threshold: int = 0, **kwargs
    ) -> None:
        """Create a F1 calculator for a certain number of classes

        Args:
            nr_classes (int, optional): Number of classes to use. Defaults to 5.
        """
        self.nr_classes = nr_classes
        self.smooth = 1e-12
        self.discard_threshold = discard_threshold
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
            if class_ground_truth.sum() <= self.discard_threshold:
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
        for pred, gt in zip(prediction, ground_truth):
            class_f1s.append(
                self._compute_sample_metric(prediction=pred, ground_truth=gt, nan=nan)
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
        **kwargs,
    ) -> torch.Tensor:
        return np.nanmean(super().__call__(prediction, ground_truth, **kwargs))

    @property
    def is_per_class(self):
        return False


class fF1Score(F1Score):
    """Inverse Log Frequency Weighted Dice Score
    Conceptually the same as the fIoU
    """

    def _compute_metric(
        self,
        ground_truth: Union[np.ndarray, List[np.ndarray]],
        prediction: Union[np.ndarray, List[np.ndarray]],
    ) -> float:
        class_f1s = super()._compute_metric(ground_truth, prediction, np.NaN)
        class_counts = self._get_class_counts(ground_truth)
        class_weights = inverse_log_frequency(class_counts)
        return np.nansum(class_weights * class_f1s, axis=1)

    @property
    def is_per_class(self):
        return False


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
    def __init__(self, f, thresholded_metric, nr_classes: int, **kwargs) -> None:
        super().__init__(nr_classes, **kwargs)
        self.f = f
        self.thresholded_metric = thresholded_metric

    def _compare(self, predictions, labels, **kwargs):
        assert (
            len(predictions.shape) == 2
        ), f"Must be 2D tensor, but got: {predictions.shape}"
        class_metric = np.empty(self.nr_classes)
        for i in range(self.nr_classes):
            y_pred = predictions.numpy()[:, i]
            y_true = labels[:, i].numpy()
            if self.thresholded_metric:
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
            thresholded_metric=True,
            nr_classes=nr_classes,
            **kwargs,
        )


class MultiLabelBalancedAccuracy(MultiLabelSklearnMetric):
    def __init__(self, nr_classes: int, **kwargs) -> None:
        super().__init__(
            f=sklearn.metrics.balanced_accuracy_score,
            thresholded_metric=True,
            nr_classes=nr_classes,
            **kwargs,
        )


class MultiLabelF1Score(MultiLabelSklearnMetric):
    def __init__(self, nr_classes: int, **kwargs) -> None:
        super().__init__(
            f=sklearn.metrics.f1_score,
            thresholded_metric=True,
            nr_classes=nr_classes,
            **kwargs,
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
        **kwargs,
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


def sum_up_gleason(annotation, n_class=4, thres=0, wsi_fix=False):
    # read the mask and count the grades
    grade_count = fast_histogram(annotation.flatten(), n_class)
    grade_count = grade_count / grade_count.sum()
    grade_count[grade_count < thres] = 0

    if wsi_fix and sum(grade_count[1:] != 0) > 1:
        grade_count[0] = 0

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
    def __init__(
        self,
        f,
        nr_classes: int,
        background_label: int,
        threshold: float = 0.25,
        callbacks=[],
        wsi_fix=False,
        **kwargs,
    ) -> None:
        """Create a IoU calculator for a certain number of classes

        Args:
            nr_classes (int, optional): Number of classes to use
        """
        self.nr_classes = nr_classes
        self.background_label = background_label
        self.f = f
        self.threshold = threshold
        self.callbacks = callbacks
        self.enabled_callbacks = False
        self.wsi_fix = wsi_fix
        super().__init__(**kwargs)

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth=None,
        tissue_mask=None,
        image_labels=None,
        **kwargs,
    ) -> Any:
        assert len(prediction) == len(ground_truth)
        assert (
            len(prediction[0].shape) == 2
        ), f"Expected 2D predictions, but got {len(prediction[0].shape)}: {prediction}"
        assert tissue_mask is None or len(tissue_mask) == len(prediction)
        assert (
            ground_truth is not None or image_labels is not None
        ), f"Must pass either ground truth or image_labels"

        gleason_grade_ground_truth = list()
        gleason_grade_prediction = list()
        for i, logits in enumerate(prediction):
            if tissue_mask is not None:
                logits[~tissue_mask[i]] = self.background_label

            if image_labels is not None:
                image_label = image_labels[i]
                if isinstance(image_label, torch.Tensor):
                    image_label = image_label.detach().cpu().numpy()
                gleason_grade_ground_truth.append(gleason_summary_wsum(image_label))
            else:
                labels = ground_truth[i]
                if tissue_mask is not None:
                    labels[~tissue_mask[i]] = self.background_label
                gleason_grade_ground_truth.append(
                    sum_up_gleason(
                        labels,
                        n_class=self.nr_classes,
                        thres=self.threshold if self.wsi_fix else 0.0,
                        wsi_fix=self.wsi_fix,
                    )
                )
            gleason_grade_prediction.append(
                sum_up_gleason(
                    logits,
                    n_class=self.nr_classes,
                    thres=self.threshold,
                    wsi_fix=self.wsi_fix,
                )
            )
        if self.enabled_callbacks:
            for callback in self.callbacks:
                callback(
                    prediction=gleason_grade_prediction,
                    ground_truth=gleason_grade_ground_truth,
                )
        return self.f(
            gleason_grade_ground_truth,
            gleason_grade_prediction,
        )

    @staticmethod
    def is_better(value: Any, comparison: Any) -> bool:
        return value >= comparison


class GleasonScoreKappa(GleasonScoreMetric):
    def __init__(self, nr_classes: int, background_label: int, **kwargs) -> None:
        super().__init__(
            f=partial(sklearn.metrics.cohen_kappa_score, weights="quadratic"),
            nr_classes=nr_classes,
            background_label=background_label,
            **kwargs,
        )
        self.enabled_callbacks = True


class GleasonScoreF1(GleasonScoreMetric):
    def __init__(self, nr_classes: int, background_label: int, **kwargs) -> None:
        super().__init__(
            f=partial(sklearn.metrics.f1_score, average="weighted"),
            nr_classes=nr_classes,
            background_label=background_label,
            **kwargs,
        )


GG_SUM_TO_LABEL = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


def assign_group(primary, secondary):
    def assign(a, b):
        if (a > 0) and (b == 0):
            b = a
        if (b > 0) and (a == 0):
            a = b
        return a, b

    if isinstance(primary, int) and isinstance(secondary, int):
        a, b = assign(primary, secondary)
        return GG_SUM_TO_LABEL[a + b]
    else:
        gg = []
        for a, b in zip(primary, secondary):
            a, b = assign(a, b)
            gg.append(GG_SUM_TO_LABEL[a + b])
        return np.array(gg)


def gleason_summary_wsum(y_pred, thres=None, wsi_fix=False):
    assert len(y_pred) == 4, f"Expected length 4, but got {len(y_pred)}: {y_pred}"

    gleason_scores = y_pred.copy()
    # remove outlier predictions
    if thres is not None:
        gleason_scores[gleason_scores < thres] = 0
    
    if wsi_fix and sum(gleason_scores[1:] != 0) > 1:
        gleason_scores[0] = 0

    # and assign overall grade
    idx = np.argsort(gleason_scores)[::-1]
    primary_class = int(idx[0])
    secondary_class = int(idx[1]) if gleason_scores[idx[1]] > 0 else int(idx[0])
    final_class = assign_group(primary_class, secondary_class)
    return final_class


class GraphClassificationGleasonScore(Metric):
    def __init__(
        self,
        f,
        nr_classes: int,
        background_label: int,
        threshold: float = 0.25,
        wsi_fix=False,
        callbacks=[],
        **kwargs,
    ) -> None:
        """Create a IoU calculator for a certain number of classes

        Args:
            nr_classes (int, optional): Number of classes to use
        """
        self.nr_classes = nr_classes
        self.background_label = background_label
        self.f = f
        self.threshold = threshold
        self.wsi_fix = wsi_fix
        self.callbacks = callbacks
        self.enabled_callbacks = False
        super().__init__(**kwargs)

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        tissue_mask=None,
        **kwargs,
    ) -> Any:
        prediction = torch.sigmoid(prediction).detach().cpu().numpy()
        gleason_grade_ground_truth = list()
        gleason_grade_prediction = list()
        for i, (logits, labels) in enumerate(zip(prediction, ground_truth)):
            if tissue_mask is not None:
                logits[~tissue_mask[i]] = self.background_label

            gleason_grade_ground_truth.append(gleason_summary_wsum(labels.numpy()))
            gleason_grade_prediction.append(
                gleason_summary_wsum(logits, thres=self.threshold, wsi_fix=self.wsi_fix)
            )
        if self.enabled_callbacks:
            for callback in self.callbacks:
                callback(
                    prediction=gleason_grade_prediction,
                    ground_truth=gleason_grade_ground_truth,
                )
        return self.f(
            gleason_grade_ground_truth,
            gleason_grade_prediction,
        )

    @staticmethod
    def is_better(value: Any, comparison: Any) -> bool:
        return value >= comparison

    @property
    def logs_model(self):
        return False


class GraphClassificationGleasonScoreKappa(GraphClassificationGleasonScore):
    def __init__(self, nr_classes: int, background_label: int, **kwargs) -> None:
        super().__init__(
            f=partial(sklearn.metrics.cohen_kappa_score, weights="quadratic"),
            nr_classes=nr_classes,
            background_label=background_label,
            **kwargs,
        )
        self.enabled_callbacks = True


class GraphClassificationGleasonScoreF1(GraphClassificationGleasonScore):
    def __init__(self, nr_classes: int, background_label: int, **kwargs) -> None:
        super().__init__(
            f=partial(sklearn.metrics.f1_score, average="weighted"),
            nr_classes=nr_classes,
            background_label=background_label,
            **kwargs,
        )


class AreaBasedGleasonScoreMetric(Metric):
    def __init__(
        self,
        f,
        callbacks=[],
        wsi_fix=False,
        threshold=0.25,
        **kwargs,
    ) -> None:
        """Create a IoU calculator for a certain number of classes

        Args:
            nr_classes (int, optional): Number of classes to use
        """
        self.f = f
        self.callbacks = callbacks
        self.wsi_fix = wsi_fix
        self.enabled_callbacks = False
        self.threshold=threshold,
        super().__init__(**kwargs)

    def __call__(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        image_labels: np.ndarray,
        **kwargs,
    ) -> Any:
        assert len(prediction) == len(image_labels)
        gleason_grade_ground_truth = list()
        gleason_grade_prediction = list()
        for predicted_area, actual_area in zip(prediction, image_labels):
            if isinstance(actual_area, torch.Tensor):
                actual_area = actual_area.detach().cpu().numpy()
            if isinstance(predicted_area, torch.Tensor):
                predicted_area = predicted_area.detach().cpu().numpy()
            gleason_grade_ground_truth.append(gleason_summary_wsum(actual_area))
            gleason_grade_prediction.append(
                gleason_summary_wsum(predicted_area, thres=self.threshold, wsi_fix=self.wsi_fix)
            )

        if self.enabled_callbacks:
            for callback in self.callbacks:
                callback(
                    prediction=gleason_grade_prediction,
                    ground_truth=gleason_grade_ground_truth,
                )
        return self.f(
            gleason_grade_ground_truth,
            gleason_grade_prediction,
        )

    @staticmethod
    def is_better(value: Any, comparison: Any) -> bool:
        return value >= comparison


class AreaBasedGleasonScoreKappa(AreaBasedGleasonScoreMetric):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            f=partial(sklearn.metrics.cohen_kappa_score, weights="quadratic"),
            **kwargs,
        )
        self.enabled_callbacks = True


class AreaBasedGleasonScoreF1(AreaBasedGleasonScoreMetric):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            f=partial(sklearn.metrics.f1_score, average="weighted"),
            **kwargs,
        )
