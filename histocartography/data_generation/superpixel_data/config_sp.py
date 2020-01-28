from utils_sp import *

class Config_SP:
    def __init__(self):
        self.data_param = 'local'

        self.base_img_dir = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/super_pixel_info/test_images/'
        self.base_sp_dir = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/super_pixel_info/'

        ## Parameters
        self.base_n_segments = 1000      # number of segments for images with 'base_n_pixels' (~300x300 pixels)
        self.max_n_segments = 10000      # number of segments increases in proportion to the size of the image.
                                         # This holds the maximum number of allowed segments
        self.base_n_pixels = 100000
        print(self.base_n_segments, self.max_n_segments)

        create_directory(self.base_sp_dir)
        create_directory(self.base_sp_dir + '1_results_basic_sp/')
        create_directory(self.base_sp_dir + '2_results_main_sp/')
    #enddef
#end



