"""Unit test for complex_module.core."""
import os
import unittest
import warnings
import numpy as np
from PIL import Image
from histocartography.io.wsi import WSI
from histocartography.io.utils import get_s3
from histocartography.io.utils import download_file_to_local


class CoreTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up test."""
        warnings.simplefilter("ignore", ResourceWarning)
        os.makedirs("tmp", exist_ok=True)
        os.makedirs("tmp/patches", exist_ok=True)
        self.s3_resource = get_s3()
        self.filename = download_file_to_local(
            s3=self.s3_resource,
            bucket_name='datasets',
            s3file='prostate/biopsy_data_all/77/77.tif',
            local_name='tmp/00_biopsy.tif')

        self.annotation_file = download_file_to_local(
            s3=self.s3_resource,
            bucket_name='datasets',
            s3file='prostate/biopsy_data_all/77/77.xml',
            local_name='tmp/01_biopsy.xml')

        self.wsi = WSI(self.filename, self.annotation_file)

    def test_image_at(self):
        """Test image_at."""
        self.wsi.image_at(5)
        self.assertAlmostEqual(5, self.wsi.current_mag)

        image2_5x = self.wsi.image_at(2.5)
        self.assertAlmostEqual(2.5, self.wsi.current_mag)
        Image.fromarray(image2_5x).save("tmp/02_biopsy_2.5x.png")

    def test_tissue_mask(self):
        """Test tissue_mask_at."""
        tissue_mask = self.wsi.tissue_mask_at(2.5)
        Image.fromarray(tissue_mask).save("tmp/03_tissue_mask_2.5x.png")

    def test_annotation_mask(self):
        """Test annotation_mask_at."""
        annotation_mask = self.wsi.annotation_mask_at(2.5)
        annotation_mask = np.uint8(annotation_mask * 255 /
                                   np.max(annotation_mask))

        Image.fromarray(annotation_mask).save(
            "tmp/04_annotation_mask_2.5x.png")

    def test_patches(self):
        """Test patches"""
        patch_generator = self.wsi.patches(
            size=(256, 256), stride=(256, 256), annotations=True, mag=2.5)
        num_patches = 0
        for patch_info in patch_generator:
            loc_x, loc_y, full_x, full_y, image, labels = patch_info
            if np.max(labels) > 0:
                labels = np.uint8(labels * 255 / np.max(labels))

            imagename = "tmp/patches/_{}x{}_image.png".format(loc_x, loc_y)
            labelname = "tmp/patches/_{}x{}_labels.png".format(loc_x, loc_y)
            Image.fromarray(image).save(imagename)
            Image.fromarray(labels).save(labelname)
            num_patches += 1

        print("Total number of patches: {}".format(num_patches))

    def tearDown(self):
        """Tear down the tests."""
        self.wsi.stack.close()
        pass
