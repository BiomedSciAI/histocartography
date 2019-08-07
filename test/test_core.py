"""Unit test for module."""
import unittest
from PIL import Image
from histocartography.io.wsi import load
from histocartography.io.utils import download_file_to_local
from histocartography.preprocessing.normalization import staining_normalization
from histocartography.preprocessing.normalization import get_mask

class ModuleTestCase(unittest.TestCase):
    """ModuleTestCase class."""

    def setUp(self):
        """Setting up the test."""
        pass

    def test_small_pipeline(self):
        """Test small pipeline combining IO and Preprocessing."""
        filename = download_file_to_local()
        image = load(wsi_file= filename, desired_level='5x')
        normalized_image = staining_normalization(image)
        mask = get_mask(normalized_image)

        self.assertEqual(image.shape[0], normalized_image.shape[0])
        self.assertEqual(image.shape[1], normalized_image.shape[1])
        self.assertEqual(image.shape[0], mask.shape[0])
        self.assertEqual(image.shape[1], mask.shape[1])
        
        

    def tearDown(self):
        """Tear down the tests."""
        pass
