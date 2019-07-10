"""Unit test for module."""
import unittest
from histocartography.core import tumor_classification_pipeline
from histocartography.ml.tumor_slide_classification import TumorSlideClassifier

class ModuleTestCase(unittest.TestCase):
    """ModuleTestCase class."""

    def setUp(self):
        """Setting up the test."""
        pass

    def test_tumor_classification_pipeline(self):
        """Test tumor_classification_pipeline()."""
        files = [1, 2, 3]
        classifier = TumorSlideClassifier()
        self.assertEqual(tumor_classification_pipeline(files, classifier), 'tumor')

    def tearDown(self):
        """Tear down the tests."""
        pass
