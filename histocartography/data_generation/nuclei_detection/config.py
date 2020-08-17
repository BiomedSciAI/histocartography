import tensorflow as tf
#from histocartography.data_generation.nuclei_detection.utils import *
from utils import *


class Config(object):
    def __init__(self, args):
        self.data_param = args.data_param
        self.tumor_type = args.tumor_type
        self.n_chunks = args.n_chunks
        self.chunk_id = args.chunk_id
        self.gpu = args.gpu
        self.type_classification = eval(args.type_classification)

        if self.data_param == 'local':
            self.base_path = '/Users/pus/Desktop/Projects/Data/Histocartography/'
            self.base_image_dir = self.base_path + \
                'PASCALE/Images_norm/' + self.tumor_type + '/'
            self.inf_model_path = self.base_path + \
                'PASCALE/misc_utils/nuclei_detection/hover_seg_Kumar.npz'
            self.base_nuclei_dir = self.base_path + 'PASCALE_NEW/nuclei_info/'

        elif self.data_param == 'dataT':
            self.base_path = '/dataT/pus/histocartography/Data/'

            #self.base_image_dir = self.base_path + 'BRACS_S/Images_norm/' + self.tumor_type + '/'
            self.base_image_dir = self.base_path + \
                'BRACS_L/Images_norm/' + self.tumor_type + '/'
            #self.base_image_dir = self.base_path + 'BACH/train/Images_norm/' + self.tumor_type + '/'
            #self.base_image_dir = self.base_path + 'BACH/test/Images_norm/'

            self.inf_model_path = self.base_path + \
                'BRACS_L/misc_utils/nuclei_detection/hover_seg_Kumar.npz'

            #self.base_nuclei_dir = self.base_path + 'BRACS_S/nuclei_info/'
            self.base_nuclei_dir = self.base_path + 'BRACS_L/nuclei_info/'
            #self.base_nuclei_dir = self.base_path + 'BACH/train/nuclei_info/'
            #self.base_nuclei_dir = self.base_path + 'BACH/test/nuclei_info/'
        # endif

        # Create directories
        create_directory(self.base_nuclei_dir)
        self.nuclei_detected_path = self.base_nuclei_dir + 'nuclei_detected/'
        create_directory(self.nuclei_detected_path)
        create_directory(self.nuclei_detected_path + 'centroids/')
        create_directory(self.nuclei_detected_path + 'instance_map/')
        self.centroid_output_dir = self.nuclei_detected_path + \
            'centroids/' + self.tumor_type + '/'
        self.map_output_dir = self.nuclei_detected_path + \
            'instance_map/' + self.tumor_type + '/'

        create_directory(self.centroid_output_dir)
        create_directory(self.map_output_dir)

        self.seed = 10
        self.model_type = 'np_hv'

        self.nr_types = 5  # denotes number of classes for nuclear type classification
        self.nr_classes = 2  # Nuclei Pixels vs Background
        self.input_norm = True  # normalize RGB to 0-1 range

        exp_id = 'v1.0/'
        model_id = '%s' % self.model_type
        self.model_name = '%s/%s' % (exp_id, model_id)

        self.log_path = './logs/'  # log root path - modify according to needs
        self.save_dir = '%s/%s' % (self.log_path,
                                   self.model_name)  # log file destination
        self.train_input_shape = [270, 270]
        self.train_mask_shape = [80, 80]

        # output will have channel ordering as [Nuclei Type][Nuclei Pixels][Additional]
        # where [Nuclei Type] will be used for getting the type of each instance
        # while [Nuclei Pixels][Additional] will be used for extracting
        # instances

        self.inf_imgs_ext = '.png'

        # Testing parameters
        self.infer_input_shape = [270, 270]
        self.infer_mask_shape = [80, 80]
        self.inf_batch_size = 4
        self.optimizer = tf.train.AdamOptimizer
        self.learning_rate = 1.0e-5

        self.eval_inf_input_tensor_names = ['images']
        self.eval_inf_output_tensor_names = ['predmap-coded']
    # enddef
# end
