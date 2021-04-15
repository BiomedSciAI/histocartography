from functools import partial
import logging
from abc import abstractmethod
from typing import Any, List, Union

import numpy as np
import sklearn.metrics
import torch


def fast_confusion_matrix(y_true: Union[np.ndarray,
                                        torch.Tensor],
                          y_pred: Union[np.ndarray,
                                        torch.Tensor],
                          nr_classes: int):
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
        y = torch.cat(
            (y,
             torch.zeros(
                 nr_classes *
                 nr_classes -
                 len(y),
                 dtype=torch.long)))
    y = y.reshape(nr_classes, nr_classes)
    return y.numpy()


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


class ConfusionMatrixMetric(Metric):
    def __init__(
            self,
            nr_classes: int,
            background_label: int,
            **kwargs) -> None:
        self.nr_classes = nr_classes
        self.background_label = background_label
        super().__init__(**kwargs)

    def _aggregate(self, confusion_matrix):
        return confusion_matrix

    def __call__(
        self,
        prediction: Union[torch.Tensor, np.ndarray],
        ground_truth: Union[torch.Tensor, np.ndarray],
        tissue_mask: Union[torch.Tensor, np.ndarray] = None,
        **kwargs,
    ) -> Any:
        """
        Compute confusion matrix.

        Args:
            prediction (Union[torch.Tensor, np.ndarray]): List of pixel-level predictions.
            ground_truth (Union[torch.Tensor, np.ndarray]): List of pixel-level ground truth
            tissue_mask (Union[torch.Tensor, np.ndarray]): List of tissue masks. Default to None.
        """
        assert len(ground_truth) == len(prediction)

        confusion_matrix = np.zeros(
            (self.nr_classes, self.nr_classes), dtype=np.int64)
        for i, (sample_gt, sample_pred) in enumerate(
                zip(ground_truth, prediction)):
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


class Dice(ConfusionMatrixMetric):
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
            scores[i] = 2 * 1 / (1 / (recall + self.smooth) +
                                 1 / (precision + self.smooth) + self.smooth)
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


class IoU(ConfusionMatrixMetric):
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


class MeanDice(Dice):
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
