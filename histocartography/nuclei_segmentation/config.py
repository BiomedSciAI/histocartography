import importlib
import tensorflow as tf

#### 
class Config(object):
    def __init__(self):

        self.seed = 10
        self.model_type = 'np_hv'

        self.type_classification = False
        self.nr_types = 5  # denotes number of classes for nuclear type classification

        self.nr_classes = 2 # Nuclei Pixels vs Background

        self.input_norm  = True # normalize RGB to 0-1 range

        ####
        exp_id = 'v1.0/'
        model_id = '%s' % self.model_type
        self.model_name = '%s/%s' % (exp_id, model_id)

        self.log_path = '/home/ubuntu/consep_pred/GraphsZRL/histocartography/histocartography/nuclei_segmentation/logs/' # log root path - modify according to needs
        self.save_dir = '%s/%s' % (self.log_path, self.model_name) # log file destination
        self.train_input_shape = [270, 270]
        self.train_mask_shape = [80, 80]


        #### Info for running inference
        # path to checkpoints will be used for inference, replace accordingly
        self.inf_model_path = '/home/ubuntu/consep_pred/Kumar/hover_seg_Kumar.npz'#'/home/ubuntu/consep_pred/hover_net/src/hover_seg_CoNSeP.npz'#'/home/ubuntu/consep_pred/Kumar/hover_seg_Kumar.npz'##self.save_dir + '/model-19640.index'

        # output will have channel ordering as [Nuclei Type][Nuclei Pixels][Additional]
        # where [Nuclei Type] will be used for getting the type of each instance
        # while [Nuclei Pixels][Additional] will be used for extracting instances

        self.inf_imgs_ext = '.png'
        self.inf_data_dir = '/home/ubuntu/consep_pred/Kumar/Images/'#'#/home/ubuntu/consep_pred/Kumar/Images/'#'/home/ubuntu/consep_pred/CoNSeP/Test/Images/'
        self.inf_output_dir = 'output/%s/%s/' % (exp_id, model_id)

        # Testing parameters
        self.infer_input_shape = [270, 270]
        self.infer_mask_shape = [80, 80]
        self.inf_batch_size = 4
        self.optimizer = tf.train.AdamOptimizer
        self.learning_rate = 1.0e-5

        # for inference during evalutaion mode i.e run by infer.py
        self.eval_inf_input_tensor_names = ['images']
        self.eval_inf_output_tensor_names = ['predmap-coded']

    def get_model(self):

        model_constructor = importlib.import_module('model.hovernet_nuclei_segmentation')
        model_constructor = model_constructor.Model_NP_HV
        return model_constructor
