"""Unit test for preprocessing.graph_builders"""
import unittest
import numpy as np
import yaml
import os
import shutil
import random

from histocartography import PipelineRunner
from histocartography.preprocessing import AssignmnentMatrixBuilder
from histocartography.utils import download_test_data


class GraphBuilderTestCase(unittest.TestCase):
    """GraphBuilderTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        self.out_path = os.path.join(self.data_path, 'assignment_matrix_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)

    def test_assignment_matrix(self):
        """
        Test assignment matrix with dummy centroids and tissue map.
        """

        # 1. define dummy data: nuclei centroids and tissue map
        image_size = (1000, 1000)
        nr_nuclei = 100
        nuclei_centroids = [[random.randint(0, image_size[0]-1), random.randint(0, image_size[1]-1)] for _ in range(nr_nuclei)]
        nuclei_centroids = np.asarray(nuclei_centroids)
        tissue_map = np.asarray([np.tile(np.expand_dims(np.array([i]*100), axis=1), 100) for i in range(1, 101)])
        tissue_map = np.squeeze(tissue_map.reshape((1, 1000, 1000)), axis=0)

        # 2. build assignment matrix 
        builder = AssignmnentMatrixBuilder()
        assignment_matrix = builder.process(nuclei_centroids, tissue_map)

        self.assertTrue(
            isinstance(
                assignment_matrix,
                np.ndarray))  # check type
        self.assertEqual(
            assignment_matrix.shape[0],
            100)   # check number of nuclei
        self.assertEqual(
            assignment_matrix.shape[1],
            100)    # check number of superpixels
        # check all nuclei assigned to only one superpixel
        self.assertEqual(np.all(np.sum(assignment_matrix, axis=1) == 1), True)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":

    unittest.main()
