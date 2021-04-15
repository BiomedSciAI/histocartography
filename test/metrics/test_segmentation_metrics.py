"""Unit test for metrics"""
import unittest
import numpy as np
import cv2
import torch
import dgl
import os
from PIL import Image
import shutil

from histocartography import PipelineRunner
from histocartography.metrics import IoU, Dice, MeanIoU, MeanDice


class SegmentationMetricsTestCase(unittest.TestCase):
    """SegmentationMetricsTestCase class."""

    def test_dice_computation(self):
        """
        Test Dice score computation.
        """

        num_classes = 5
        pred = [np.random.randint(num_classes, size=(100, 100))
                for _ in range(10)]
        gt = [np.random.randint(num_classes, size=(100, 100))
              for _ in range(10)]

        evaluator = Dice(nr_classes=num_classes, background_label=0)
        out = evaluator(pred, gt)

        self.assertIsInstance(out, np.ndarray)  # output is numpy
        # shape is number of classes
        self.assertEqual(out.shape[0], num_classes)
        for entry in out:
            self.assertLessEqual(entry, 1.)
        for entry in out:
            self.assertGreaterEqual(entry, 0.)

        out = evaluator(pred, pred)
        for entry in out[1:]:
            self.assertAlmostEqual(entry, 1.)

    def test_dice_computation_with_tissue_mask(self):
        """
        Test Dice score computation with tissue mask.
        """

        num_classes = 5
        pred = [np.random.randint(num_classes, size=(100, 100))
                for _ in range(10)]
        gt = [np.random.randint(num_classes, size=(100, 100))
              for _ in range(10)]
        tissue_mask = [
            np.random.randint(
                2, size=(
                    100, 100)) for _ in range(10)]

        evaluator = Dice(nr_classes=num_classes, background_label=0)
        out = evaluator(pred, gt, tissue_mask=tissue_mask)

        self.assertIsInstance(out, np.ndarray)  # output is numpy
        # shape is number of classes
        self.assertEqual(out.shape[0], num_classes)
        for entry in out:
            self.assertLessEqual(entry, 1.)  # always less than 1
        for entry in out:
            self.assertGreaterEqual(entry, 0.)  # but always greater than 0

        out = evaluator(pred, pred, tissue_mask=tissue_mask)
        for entry in out[1:]:
            self.assertAlmostEqual(entry, 1.)

    def test_iou_computation(self):
        """
        Test IoU score computation.
        """

        num_classes = 5
        pred = [np.random.randint(num_classes, size=(100, 100))
                for _ in range(10)]
        gt = [np.random.randint(num_classes, size=(100, 100))
              for _ in range(10)]

        evaluator = IoU(nr_classes=num_classes, background_label=0)
        out = evaluator(pred, gt)

        self.assertIsInstance(out, np.ndarray)  # output is numpy
        # shape is number of classes
        self.assertEqual(out.shape[0], num_classes)
        for entry in out:
            self.assertLessEqual(entry, 1.)
        for entry in out:
            self.assertGreaterEqual(entry, 0.)

        out = evaluator(pred, pred)
        for entry in out[1:]:
            self.assertAlmostEqual(entry, 1.)

    def test_iou_computation_with_tissue_mask(self):
        """
        Test IoU score computation with tissue mask.
        """

        num_classes = 5
        pred = [np.random.randint(num_classes, size=(100, 100))
                for _ in range(10)]
        gt = [np.random.randint(num_classes, size=(100, 100))
              for _ in range(10)]
        tissue_mask = [
            np.random.randint(
                2, size=(
                    100, 100)) for _ in range(10)]

        evaluator = IoU(nr_classes=num_classes, background_label=0)
        out = evaluator(pred, gt, tissue_mask=tissue_mask)

        self.assertIsInstance(out, np.ndarray)  # output is numpy
        # shape is number of classes
        self.assertEqual(out.shape[0], num_classes)
        for entry in out:
            self.assertLessEqual(entry, 1.)  # always less than 1
        for entry in out:
            self.assertGreaterEqual(entry, 0.)  # but always greater than 0

        out = evaluator(pred, pred, tissue_mask=tissue_mask)
        for entry in out[1:]:
            self.assertAlmostEqual(entry, 1.)

    def test_mean_dice_computation(self):
        """
        Test Mean Dice score computation.
        """

        num_classes = 5
        pred = [np.random.randint(num_classes, size=(100, 100))
                for _ in range(10)]
        gt = [np.random.randint(num_classes, size=(100, 100))
              for _ in range(10)]

        evaluator = MeanDice(nr_classes=num_classes, background_label=0)
        out = evaluator(pred, gt)

        self.assertIsInstance(out, float)  # output is float
        self.assertLessEqual(out, 1.)
        self.assertGreaterEqual(out, 0.)

    def test_mean_dice_computation_with_tissue_mask(self):
        """
        Test Mean Dice score computation with tissue mask.
        """

        num_classes = 5
        pred = [np.random.randint(num_classes, size=(100, 100))
                for _ in range(10)]
        gt = [np.random.randint(num_classes, size=(100, 100))
              for _ in range(10)]
        tissue_mask = [
            np.random.randint(
                2, size=(
                    100, 100)) for _ in range(10)]

        evaluator = MeanDice(nr_classes=num_classes, background_label=0)
        out = evaluator(pred, gt, tissue_mask=tissue_mask)

        self.assertIsInstance(out, float)  # output is float
        self.assertLessEqual(out, 1.)
        self.assertGreaterEqual(out, 0.)

    def test_mean_iou_computation(self):
        """
        Test MeanIoU score computation.
        """

        num_classes = 5
        pred = [np.random.randint(num_classes, size=(100, 100))
                for _ in range(10)]
        gt = [np.random.randint(num_classes, size=(100, 100))
              for _ in range(10)]

        evaluator = MeanIoU(nr_classes=num_classes, background_label=0)
        out = evaluator(pred, gt)

        self.assertIsInstance(out, float)  # output is float
        self.assertLessEqual(out, 1.)
        self.assertGreaterEqual(out, 0.)

    def test_mean_iou_computation_with_tissue_mask(self):
        """
        Test IoU score computation with tissue mask.
        """

        num_classes = 5
        pred = [np.random.randint(num_classes, size=(100, 100))
                for _ in range(10)]
        gt = [np.random.randint(num_classes, size=(100, 100))
              for _ in range(10)]
        tissue_mask = [
            np.random.randint(
                2, size=(
                    100, 100)) for _ in range(10)]

        evaluator = MeanIoU(nr_classes=num_classes, background_label=0)
        out = evaluator(pred, gt, tissue_mask=tissue_mask)

        self.assertIsInstance(out, float)  # output is float
        self.assertLessEqual(out, 1.)
        self.assertGreaterEqual(out, 0.)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    unittest.main()
