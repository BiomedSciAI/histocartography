from utils import *


class Config:
    def __init__(self, args):
        self.mode = args.mode
        self.data_param = args.data_param
        self.patch_size = args.patch_size
        self.is_mask = eval(args.is_mask)
        self.merging_type = args.merging_type

        self.encoder = args.encoder

        self.batch_size = args.batch_size
        self.info = args.info
        self.tumor_type = args.tumor_type

        # Early stopping parameters
        self.early_stopping_min_delta = 0.0001
        self.early_stopping_patience = 20

        # Reduce learning rate on plateau parameters
        self.learning_rate_decay = 0.5
        self.min_learning_rate = 0.000001
        self.reduce_lr_patience = 5

        #self.tumor_types = ['benign', 'pathologicalbenign', 'udh', 'adh', 'fea', 'dcis', 'malignant']
        self.tumor_types = [
            'benign',
            'pathologicalbenign',
            'dcis',
            'malignant']
        self.tumor_labels = [0, 1, 1, 2, 2, 3, 4]

        self.merging_name = ''
        self.features_name = ''

        if self.data_param == 'local':
            self.base_path = '/Users/pus/Desktop/Projects/Data/Histocartography/'
            self.base_img_dir = self.base_path + 'PASCALE/Images_norm/'
            self.base_sp_dir = self.base_path + 'PASCALE_NEW/super_pixel_info/'
            self.sp_classifier_path = self.base_path + \
                'PASCALE_NEW/misc_utils/merging_sp_classification/sp_classifier/'

        elif self.data_param == 'dataT':
            self.base_path = '/dataT/pus/histocartography/Data/'

            self.base_img_dir = self.base_path + 'BRACS_L/Images_norm/'
            #self.base_img_dir = self.base_path + 'BACH/train/Images_norm/'
            #self.base_img_dir = self.base_path + 'BACH/test/Images_norm/'

            self.base_sp_dir = self.base_path + 'BRACS_L/super_pixel_info/'
            #self.base_sp_dir = self.base_path + 'BACH/train/super_pixel_info/'
            #self.base_sp_dir = self.base_path + 'BACH/test/super_pixel_info/'

            self.sp_classifier_path = self.base_path + \
                'BRACS_L/misc_utils/merging_sp_classification/sp_classifier/'

        # Create directories
        create_directory(self.base_sp_dir)

        # ------------------------------------------------------------------------------------------- UNMERGED
        self.sp_unmerged_detected_path = self.base_sp_dir + 'sp_unmerged_detected/'
        create_directory(self.sp_unmerged_detected_path)
        create_directory(self.sp_unmerged_detected_path + 'instance_map/')
        create_directory(self.sp_unmerged_detected_path + 'centroids/')
        self.create_tumor_wise_folders(
            self.sp_unmerged_detected_path + 'instance_map/')
        self.create_tumor_wise_folders(
            self.sp_unmerged_detected_path + 'centroids/')

        if self.mode == 'features_cnn':
            self.get_feature_name()
            self.sp_unmerged_features_path = self.base_sp_dir + 'sp_unmerged_features/'
            create_directory(self.sp_unmerged_features_path)
            self.sp_unmerged_features_path += self.features_name + '/'
            create_directory(self.sp_unmerged_features_path)
            self.create_tumor_wise_folders(self.sp_unmerged_features_path)

        # ------------------------------------------------------------------------------------------- MERGED
        self.get_merging_name()
        self.sp_merged_detected_path = self.base_sp_dir + 'sp_merged_detected/'
        create_directory(self.sp_merged_detected_path)
        self.sp_merged_detected_path += self.merging_name + '/'
        create_directory(self.sp_merged_detected_path)
        create_directory(self.sp_merged_detected_path + 'instance_map/')
        create_directory(self.sp_merged_detected_path + 'centroids/')
        self.create_tumor_wise_folders(
            self.sp_merged_detected_path + 'instance_map/')
        self.create_tumor_wise_folders(
            self.sp_merged_detected_path + 'centroids/')

        if 'features' in self.mode:
            self.get_feature_name()
            self.sp_merged_features_path = self.base_sp_dir + 'sp_merged_features/'
            create_directory(self.sp_merged_features_path)
            self.sp_merged_features_path += self.merging_name + '_' + self.features_name + '/'
            create_directory(self.sp_merged_features_path)
            self.create_tumor_wise_folders(self.sp_merged_features_path)

        # Parameters
        # number of segments for images with 'base_n_pixels' (~300x300 pixels)
        self.base_n_segments = 1000
        # number of segments increases in proportion to the size of the image.
        self.max_n_segments = 10000
        # This holds the maximum number of allowed segments
        self.base_n_pixels = 100000
        print(
            'base_n_segments:',
            self.base_n_segments,
            'max_n_segments:',
            self.max_n_segments)
    # enddef

    def create_tumor_wise_folders(self, path):
        for t in self.tumor_types:
            create_directory(path + t)
    # enddef

    def get_feature_name(self):
        if 'features_hc' in self.mode:
            self.features_name = 'features_hc' + '_' + self.info
        elif 'features_cnn' in self.mode:
            self.features_name = 'features_cnn_' + self.encoder + \
                '_mask_' + str(self.is_mask) + '_' + self.info
    # enddef

    def get_merging_name(self):
        if self.merging_type == 'hc':
            self.merging_name = 'merging_hc'
        elif self.merging_type == 'cnn':
            self.merging_name = 'merging_cnn_' + \
                self.encoder + '_mask_' + str(self.is_mask)
    # enddef


# end
