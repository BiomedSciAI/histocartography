import importlib
import tensorflow as tf


class Config(object):
    def __init__(self):
        self.data_param = 'dataT'
        self.type_classification = False

        tumor_type = '0_benign'    # 0_benign, 1_pathological_benign, 2_udh, 3_adh, 4_fea, 5_dcis, 6_malignant

        if self.data_param == 'local':
            # Inference model path
            self.inf_model_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/nuclei_info/hover_seg_Kumar.npz'

            # Inference data path
            self.inf_data_dir = '/Users/pus/Desktop/Projects/Data/Hiastocartography/PASCALE/Images_norm/' + tumor_type + '/'

            # Inference prediction results save path
            self.inf_output_dir = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/nuclei_info/Predictions/' + tumor_type + '/'

        elif self.data_param == 'dpmiccai':
            ##### Models trained on CoNSeP data
            '''
            self.inf_data_dir = '/home/ubuntu/histocartography/CRC/Images_norm/' 
            self.inf_model_path = '/home/ubuntu/histocartography/hover_net/src/hover_seg_CoNSeP.npz'
            self.inf_output_dir = '/home/ubuntu/histocartography/feature_extraction/nuclei_segmentation/output_CRC/'
            #'''

            ##### Models trained on MoNuSeg data
            #'''
            self.inf_data_dir = '/home/ubuntu/histocartography/Data/PASCALE/Images_norm/' + tumor_type + '/'
            self.inf_model_path = '/home/ubuntu/histocartography/Data/PASCALE/nuclei_info/hover_seg_Kumar.npz'
            self.inf_output_dir = '/home/ubuntu/histocartography/Data/PASCALE/nuclei_info/Predictions/' + tumor_type + '/'
            #'''

        elif self.data_param == 'dataT':
            self.inf_data_dir = '/dataT/pus/histocartography/Data/PASCALE/Images_norm/' + tumor_type + '/'
            self.inf_model_path = '/dataT/pus/histocartography/Data/PASCALE/nuclei_info/hover_seg_Kumar.npz'
            self.inf_output_dir = '/dataT/pus/histocartography/Data/PASCALE/nuclei_info/Predictions/' + tumor_type + '/'

        elif self.data_param == 'dataL':
            self.inf_data_dir = '/dataL/pus/histocartography/Data/PASCALE/Images_norm/' + tumor_type + '/'
            self.inf_model_path = '/dataL/pus/histocartography/Data/PASCALE/nuclei_info/hover_seg_Kumar.npz'
            self.inf_output_dir = '/dataL/pus/histocartography/Data/PASCALE/nuclei_info/Predictions/' + tumor_type + '/'
        #endif

        self.seed = 10
        self.model_type = 'np_hv'

        self.nr_types = 5  # denotes number of classes for nuclear type classification
        self.nr_classes = 2  # Nuclei Pixels vs Background
        self.input_norm = True  # normalize RGB to 0-1 range

        exp_id = 'v1.0/'
        model_id = '%s' % self.model_type
        self.model_name = '%s/%s' % (exp_id, model_id)

        self.log_path = './logs/'  # log root path - modify according to needs
        self.save_dir = '%s/%s' % (self.log_path, self.model_name)  # log file destination
        self.train_input_shape = [270, 270]
        self.train_mask_shape = [80, 80]

        # output will have channel ordering as [Nuclei Type][Nuclei Pixels][Additional]
        # where [Nuclei Type] will be used for getting the type of each instance
        # while [Nuclei Pixels][Additional] will be used for extracting instances

        self.inf_imgs_ext = '.png'

        # Testing parameters
        self.infer_input_shape = [270, 270]
        self.infer_mask_shape = [80, 80]
        self.inf_batch_size = 4
        self.optimizer = tf.train.AdamOptimizer
        self.learning_rate = 1.0e-5

        self.eval_inf_input_tensor_names = ['images']
        self.eval_inf_output_tensor_names = ['predmap-coded']

    def get_model(self):

        model_constructor = importlib.import_module('model.hovernet_nuclei_segmentation')
        model_constructor = model_constructor.Model_NP_HV
        return model_constructor
