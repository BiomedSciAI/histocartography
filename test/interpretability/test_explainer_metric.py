"""Unit test for interpretability.explainer_metrics"""
import unittest
import numpy as np
import h5py
import random

from histocartography.interpretability.explainer_metrics import ExplainerMetric

NUMBER_DUMMY_SAMPLES = 100


class TestExplainerMetric(unittest.TestCase):
    """TestExplainerMetric class."""

    def setUp(self):
        """Setting up the test."""
    
    def test_histogram_construction(self):
        """Test Histogram construction.
        """

        # 1. Load all the required inputs:

        # i. load importance score
        random_num_nuclei = [random.randint(100, 500) for _ in range(NUMBER_DUMMY_SAMPLES)]
        importance_score_list = [np.random.randn(random_num_nuclei[i],) for i in range(NUMBER_DUMMY_SAMPLES)]

        # ii. load concepts 
        nuclei_concepts_list = [np.random.randn(random_num_nuclei[i], 10) for i in range(NUMBER_DUMMY_SAMPLES)]

        # iii. load TRoI labels  
        tumor_labels = [random.randint(0, 2) for _ in range(NUMBER_DUMMY_SAMPLES)]

        # 2. run the explainer  
        analyser = ExplainerMetric()
        separability_scores = analyser.process(
            importance_score_list,
            nuclei_concepts_list,
            tumor_labels
        )

        print('Separability scores:', separability_scores)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    model = TestExplainerMetric()
    model.test_histogram_construction()
