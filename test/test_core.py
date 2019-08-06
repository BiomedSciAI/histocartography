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
        #DUMMY TEST, DOESN'T TEST ANYTHING
        # TODO when the tumor classification pipeline works.
        self.assertEqual('tumor', 'tumor')

    def tearDown(self):
        """Tear down the tests."""
        pass
