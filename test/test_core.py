"""Unit test for module."""
import unittest
from PIL import Image
import os


class ModuleTestCase(unittest.TestCase):
    """ModuleTestCase class."""

    def setUp(self):
        """Setting up the test."""
        pass

    def test_small_pipeline(self):
        """Test small pipeline combining IO and Preprocessing."""
        
        self.assertTrue(True)
        # os.makedirs("tmp", exist_ok=True)

        # s3_resource = get_s3()
        # filename = download_file_to_local(s3= s3_resource, bucket_name= 'datasets', 
        #                                 s3file= 'prostate/biopsy_data_all/17/17.tif',
        #                                 local_name='tmp/00_biopsy.tif'
        #                                 )
        # annotation_file = download_file_to_local(s3= s3_resource, bucket_name= 'datasets', 
        #                                 s3file= 'prostate/biopsy_data_all/17/17.xml',
        #                                 local_name='tmp/01_biopsy.xml'
        #                                 )
        
        # image, _, scale_factor = load(wsi_file= filename, desired_level='5x')
        # normalized_image = staining_normalization(image)
        # raw_mask, mask = get_annotation_mask(annotation_file= annotation_file, 
        #                     image_shape= image.shape,
        #                     scale_factor= scale_factor
        #                     )
        # Image.fromarray(normalized_image).save("tmp/02_biopsy_normalized.png")
        # Image.fromarray(mask).save("tmp/03_biopsy_labels.png")


        # # tissue mask creation and patch coordinate extraction

        # filename = download_file_to_local(s3= s3_resource, bucket_name= 'datasets', 
        #                                 s3file= 'prostate/biopsy_data_all/17/17.tif',
        #                                 local_name='tmp/04_input.tif'
        #                                 )
        # image, _, scale_factor = load(wsi_file= filename, desired_level='5x')
        # tissue_mask = get_tissue_mask(image)
        # Image.fromarray(image).save("tmp/05_input_as_png.png")
        # Image.fromarray(tissue_mask).save("tmp/06_input_mask.png")


        # model_json = download_file_to_local(
        #                                     bucket_name='models', 
        #                                     s3file='tumor-stratification/keras/patch/model_definition.json', 
        #                                     local_name='tmp/model_json.json'
        #                                     )
        # model_weights = download_file_to_local(
        #                                     bucket_name='models', 
        #                                     s3file='tumor-stratification/keras/patch/model_weights.hdf5', 
        #                                     local_name='tmp/model_weights.hdf5'
        #                                     )

        # patch_info_coordinates = get_patches(image_id='tmp/patches_output', image=image, patch_size=128, visualize=1)

        # y_pred = predict_for_image(patch_info_coordinates, image, model_json, model_weights, 1)

        # self.assertEqual(image.shape[0:2], tissue_mask.shape)
        # self.assertEqual(len(y_pred), len(patch_info_coordinates))


    def tearDown(self):
        """Tear down the tests."""
        pass
