"""Unit test for module."""
import unittest
from PIL import Image
import os
from histocartography.io.wsi import load
from histocartography.io.utils import get_s3
from histocartography.io.utils import download_file_to_local
from histocartography.preprocessing.normalization import staining_normalization
from histocartography.io.annotations import get_annotation_mask
from histocartography.preprocessing.tissue_mask import get_tissue_mask
from histocartography.preprocessing.patch_extraction import get_patches


class ModuleTestCase(unittest.TestCase):
    """ModuleTestCase class."""

    def setUp(self):
        """Setting up the test."""
        pass

    def test_small_pipeline(self):
        """Test small pipeline combining IO and Preprocessing."""

        s3_resource = get_s3()
        filename = download_file_to_local(s3= s3_resource, bucket_name= 'datasets', 
                                        s3file= 'prostate/biopsy_data_all/17/17.tif',
                                        local_name='tmp.tif'
                                        )
        annotation_file = download_file_to_local(s3= s3_resource, bucket_name= 'datasets', 
                                        s3file= 'prostate/biopsy_data_all/17/17.xml',
                                        local_name='tmp.xml'
                                        )
        
        image, _, scale_factor = load(wsi_file= filename, desired_level='5x')
        
        normalized_image = staining_normalization(image)
        
        raw_mask, mask = get_annotation_mask(annotation_file= annotation_file, 
                            image_shape= image.shape,
                            scale_factor= scale_factor
                            )
        
        Image.fromarray(normalized_image).save("tmp_slide.png")
        Image.fromarray(mask).save("tmp_labels.png")


        # tissue mask creation and patch coordinate extraction

        filename = download_file_to_local()
        image, _, scale_factor = load(wsi_file= filename, desired_level='10x')
        tissue_mask = get_tissue_mask(image)
        Image.fromarray(image).save("s3_img.png")
        Image.fromarray(tissue_mask).save("s3_tissue_mask.png")

        patch_info_coordinates = get_patches(image_id='s3_img', image=image, patch_size=256, visualize=1)


        self.assertEqual(image.shape[0:2], tissue_mask.shape)




    def tearDown(self):
        """Tear down the tests."""
        pass
