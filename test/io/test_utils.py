"""Unit test for complex_module.core."""
import os
import unittest
from histocartography.io.utils import download_file_to_local
from histocartography.io.utils import save_local_file
from histocartography.io.utils import get_s3





class CoreTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""
        os.makedirs("tmp", exist_ok=True)
        self.s3 = get_s3()
        

    def test_download_file_to_local(self):
        """Test download_file_to_local()."""

        desired_name = 'tmp/tmp.svs'
        saved_name = download_file_to_local(s3=self.s3, local_name=desired_name)


        self.assertEqual(saved_name, desired_name)

    def test_save_local_file(self):
        """Test save_local_file()."""

        desired_name = 'tmp/tmp.svs'
        saved_name = download_file_to_local(s3=self.s3, local_name=desired_name)
        save_local_file(saved_name, s3=self.s3 , bucket_name="test-data", s3file="upload.svs")

        self.assertEqual(saved_name, desired_name)

    def tearDown(self):
        """Tear down the tests."""
        
