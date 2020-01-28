from utils_sp import *

class Config_SP:
    def __init__(self, data_param, prob_thr):
        self.data_param = data_param
        self.prob_thr = prob_thr

        if self.data_param == 'local':
            self.base_img_dir = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/Images_norm/'
            self.base_sp_dir = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/super_pixel_info/'
            self.sp_classifier_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/misc_utils/sp_classification/'

        elif self.data_param == 'dataT':
            self.base_img_dir = '/dataT/pus/histocartography/Data/PASCALE/Images_norm/'
            self.base_sp_dir = '/dataT/pus/histocartography/Data/PASCALE/super_pixel_info/'
            self.sp_classifier_path = '/dataT/pus/histocartography/Data/PASCALE/misc_utils/sp_classification/'

        create_directory(self.base_sp_dir)

        self.basic_sp_path = self.base_sp_dir + 'basic_sp/'
        create_directory(self.basic_sp_path)

        self.main_sp_path = self.base_sp_dir + 'main_sp/'
        create_directory(self.main_sp_path)
        self.main_sp_path += 'prob_thr_' + str(prob_thr) + '/'
        create_directory(self.main_sp_path)

        self.sp_img_path = self.base_sp_dir + 'sp_image/'
        create_directory(self.sp_img_path)
        create_directory(self.sp_img_path + 'basic_sp/')
        create_directory(self.sp_img_path + 'main_sp/')
        create_directory(self.sp_img_path + 'main_sp/prob_thr_' + str(prob_thr) + '/')

        self.tumor_types = ['benign', 'pathologicalbenign', 'udh', 'adh', 'fea', 'dcis', 'malignant']

        for tt in self.tumor_types:
            create_directory(self.basic_sp_path + tt + '/')
            create_directory(self.main_sp_path + tt + '/')
            create_directory(self.sp_img_path + 'basic_sp/' + tt + '/')
            create_directory(self.sp_img_path + 'main_sp/prob_thr_' + str(prob_thr) + '/' + tt + '/')

        ## Parameters
        self.base_n_segments = 1000      # number of segments for images with 'base_n_pixels' (~300x300 pixels)
        self.max_n_segments = 10000      # number of segments increases in proportion to the size of the image.
                                         # This holds the maximum number of allowed segments
        self.base_n_pixels = 100000
        print('prob_thr:', prob_thr, 'base_n_segments:', self.base_n_segments, 'max_n_segments:', self.max_n_segments)

    #enddef
#end



