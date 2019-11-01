"""Unit test for complex_module.core."""
import os
import unittest
import warnings
import numpy as np
from PIL import Image
from histocartography.io.wsi import WSI
from histocartography.io.annotations import XMLAnnotation
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
            local_name='tmp/00_biopsy.tif'
        )

        self.annotation_file = download_file_to_local(
            s3=self.s3_resource,
            bucket_name='datasets',
            s3file='prostate/biopsy_data_all/77/77.xml',
            local_name='tmp/01_biopsy.xml'
        )
        annotations = XMLAnnotation(
            self.annotation_file,
            ['background', 'NROI', '3+3', '3+4', '4+3', '4+4', '4+5', '5+5']
        )

        #self.wsi = WSI(self.filename, annotations)
        self.wsi = WSI('/Users/fra/Downloads/tumor_105.tif', annotations)
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

    def test_patches(self):
        """Test patches"""
        patch_generator = self.wsi.patches(
            size=(1024, 1024), stride=(512, 512), annotations=True, mag=1
        )
        num_patches = 0
        for patch in patch_generator:
            loc_x, loc_y, full_x, full_y, x_mag, y_mag, image, labels = patch
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
