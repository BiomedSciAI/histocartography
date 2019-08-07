"""Unit test for complex_module.core."""
import unittest
from histocartography.io.wsi import load
from histocartography.io.wsi import save
from histocartography.io.wsi import patch
from histocartography.io.utils import download_file_to_local


class CoreTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""
        pass

    def test_load(self):
        """Test load()."""

        filename = download_file_to_local()

        image10x, scale_factor_10x = load(filename)
        image5x, scale_factor_5x = load(filename,desired_level="5x")


        self.assertAlmostEqual(image10x.shape[0], 2*image5x.shape[0])
        self.assertAlmostEqual(scale_factor_10x*2 , scale_factor_5x, places=1)

    def test_save(self):
        """Test save()."""
        self.assertEqual(save(), 'Save')

    def test_patch(self):
        """Test patch()."""
        self.assertEqual(patch(), 'Patch')

    def tearDown(self):
        """Tear down the tests."""
        pass
