"""Unit test for preprocessing.graph_builders"""
import unittest
import numpy as np
import yaml
import os
import shutil

from histocartography import PipelineRunner
from histocartography.utils.io import download_test_data


class GraphBuilderTestCase(unittest.TestCase):
    """GraphBuilderTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        download_test_data(self.data_path)
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '283_dcis_4.png'
        self.out_path = os.path.join(self.data_path, 'assignment_matrix_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path)
        os.makedirs(self.out_path)

    def test_assignment_matrix_with_pipeline_runner(self):
        """
        Test assignment matrix with pipeline runner.
        """

        config_fname = os.path.join(self.current_path, 'config', 'assignment_matrix', 'assignment_matrix_builder.yml')
        with open(config_fname, 'r') as file:
            config = yaml.load(file)

        pipeline = PipelineRunner(output_path=self.out_path, **config)
        output = pipeline.run(
            name=self.image_name.replace('.png', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        assignment_matrix = output['assignment_matrix']

        self.assertTrue(isinstance(assignment_matrix, np.ndarray))  # check type
        self.assertEqual(assignment_matrix.shape[0], 331)   # check number of nuclei
        self.assertEqual(assignment_matrix.shape[1], 23)    # check number of superpixels
        self.assertEqual(np.all(np.sum(assignment_matrix, axis=1) == 1), True)  # check all nuclei assigned to only one superpixel

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":

    unittest.main()
