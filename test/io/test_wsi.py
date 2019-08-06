"""Unit test for complex_module.core."""
import unittest
from histocartography.io.wsi import load
from histocartography.io.wsi import save
from histocartography.io.wsi import patch



class CoreTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""
        pass

    def test_load(self):
        """Test load()."""
        image10x = load("test/data/wsi_test.tif")
        image5x = load("test/data/wsi_test.tif",desired_level="5x")

        self.assertEqual(image10x.shape[0],2*image5x.shape[0])

    def test_save(self):
        """Test save()."""
        self.assertEqual(save(), 'Save')

    def test_patch(self):
        """Test patch()."""
        self.assertEqual(patch(), 'Patch')

    def tearDown(self):
        """Tear down the tests."""
        pass
