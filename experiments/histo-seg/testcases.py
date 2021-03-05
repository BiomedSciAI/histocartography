import unittest
import torch
from sklearn.metrics import jaccard_score, f1_score
from metrics import IoU, F1Score
import numpy as np


class IoUTestCase(unittest.TestCase):
    def test_iou(self):
        batch_size = 10
        x = 100
        y = 100

        r1 = torch.randint(0, 4, size=(batch_size, x, y))
        r2 = torch.randint(0, 4, size=(batch_size, x, y))

        for i in range(r2.shape[0]):
            r2[i][r2[i] == torch.randint(0, 4, (1,))] = torch.randint(0, 4, (1,))

        def _sklearn_method(prediction, ground_truth):
            class_f1 = np.empty(4)
            for class_label in range(4):
                class_ground_truth = ground_truth == class_label
                if not class_ground_truth.any():
                    class_f1[class_label] = np.NaN
                    continue
                class_prediction = prediction == class_label
                class_f1[class_label] = jaccard_score(
                    y_pred=class_prediction.ravel(), y_true=class_ground_truth.ravel()
                )
            return class_f1

        metric = IoU(4, 4)

        my_result = metric(prediction=r1.numpy(), ground_truth=r2.numpy())

        sklearn_result = list()
        for i in range(batch_size):
            sklearn_result.append(
                _sklearn_method(prediction=r1[i].numpy(), ground_truth=r2[i].numpy())
            )
        sklearn_result = np.nanmean(np.stack(sklearn_result), axis=0)

        self.assertEqual(sklearn_result.shape, my_result.shape)
        self.assertTrue(
            all((sklearn_result == sklearn_result) == (my_result == my_result))
        )
        self.assertTrue(
            np.nanmax(np.abs(sklearn_result.ravel() - my_result.ravel())) < 1e-8
        )


class F1TestCase(unittest.TestCase):
    def test_iou(self):
        batch_size = 10
        x = 100
        y = 100

        r1 = torch.randint(0, 4, size=(batch_size, x, y))
        r2 = torch.randint(0, 4, size=(batch_size, x, y))

        for i in range(r2.shape[0]):
            r2[i][r2[i] == torch.randint(0, 4, (1,))] = torch.randint(0, 4, (1,))

        def _sklearn_method(prediction, ground_truth):
            class_f1 = np.empty(4)
            for class_label in range(4):
                class_ground_truth = ground_truth == class_label
                if not class_ground_truth.any():
                    class_f1[class_label] = np.NaN
                    continue
                class_prediction = prediction == class_label
                class_f1[class_label] = f1_score(
                    y_pred=class_prediction.ravel(), y_true=class_ground_truth.ravel()
                )
            return class_f1

        metric = F1Score(4)

        my_result = metric(prediction=r1.numpy(), ground_truth=r2.numpy())

        sklearn_result = list()
        for i in range(batch_size):
            sklearn_result.append(
                _sklearn_method(prediction=r1[i].numpy(), ground_truth=r2[i].numpy())
            )
        sklearn_result = np.nanmean(np.stack(sklearn_result), axis=0)

        self.assertEqual(sklearn_result.shape, my_result.shape)
        self.assertTrue(
            all((sklearn_result == sklearn_result) == (my_result == my_result))
        )
        self.assertTrue(
            np.nanmax(np.abs(sklearn_result.ravel() - my_result.ravel())) < 1e-8
        )


if __name__ == "__main__":
    unittest.main()