"""Unit test for interpretability.explainer_metrics"""
import unittest
import numpy as np
import h5py
import random
import numpy as np

from histocartography.interpretability.explainer_metrics import ExplainerMetric, ExplainerMetricAnalyser

NUMBER_DUMMY_SAMPLES = 100

PATH_PRIOR = np.asarray([
    [0.617283951, 0.672131148, 0.529069767],
    [0.839506173, 0.896174863, 0.872093023],
    [1., 1., 1.],
    [0., 0., 0.],
    [0.25308642, 0.404371585, 0.377906977],
])

RISK = np.array([1, 2, 1])

CONCEPT_GROUPING = {
    'size': ['area'],
    'shape': ['perimeter', 'roughness', 'eccentricity', 'circularity'],
    'shape_variation': ['shape_factor'],
    'spacing': ['mean_crowdedness', 'std_crowdedness'],
    'chromatin': ['dissimilarity', 'contrast', 'homogeneity', 'asm', 'entropy', 'variance']
}

NUMBER_OF_CONCEPTS = sum([len(v) for _, v in CONCEPT_GROUPING.items()])


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
        nuclei_concepts_list = [np.random.randn(random_num_nuclei[i], NUMBER_OF_CONCEPTS) for i in range(NUMBER_DUMMY_SAMPLES)]

        # iii. load TRoI labels  
        tumor_labels = [random.randint(0, 2) for _ in range(NUMBER_DUMMY_SAMPLES)]

        # 2. compute separability score matrix 
        metric_explainer = ExplainerMetric()
        separability_scores = metric_explainer.process(
            importance_score_list,
            nuclei_concepts_list,
            tumor_labels
        )

        print('Separability scores:', separability_scores)

        # 3. aggregate, compute and group score 
        analyser = ExplainerMetricAnalyser(
            separability_scores=separability_scores,
            concept_grouping=CONCEPT_GROUPING,
            risk=RISK,
            path_prior=PATH_PRIOR    
        )
        max_separability = analyser.compute_max_separability_score()
        average_separability = analyser.compute_average_separability_score()
        correlation_separability = analyser.compute_correlation_separability_score()

        print('Max separability:', max_separability)
        print('Average separability:', average_separability)
        print('Correlation separability:', correlation_separability)

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    model = TestExplainerMetric()
    model.test_histogram_construction()
