"""Unit test for preprocessing.io"""
import unittest
import hashlib
import os

from histocartography.utils import download_box_link


class IOTestCase(unittest.TestCase):
    """IOTestCase class."""

    @classmethod
    def setUpClass(self):
        self.box_test_url = 'https://ibm.box.com/shared/static/30uzamx0xr222waqc2dx3uptnvcb5jf0.bmp'
        self.box_md5 = 'd9002fd4dce81f0246626f1df38fdf26'
        self.box_file = 'noise.bmp'

    def test_download_box_link(self):
        """
        Test downloading a large file from box.
        """
        local_file = download_box_link(self.box_test_url, self.box_file)
        with open(local_file, "rb") as file_to_check:
            # read contents of the file
            data = file_to_check.read()
            # pipe contents of the file through
            local_hash = hashlib.md5(data).hexdigest()

        # Check that file was correctly downloaded
        self.assertEqual(local_hash, self.box_md5)

    def tearDown(self):
        """Tear down the tests."""
        if os.path.exists(self.box_file):
            os.remove(self.box_file)


if __name__ == "__main__":
    unittest.main()
