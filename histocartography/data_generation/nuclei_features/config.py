import torch
from utils import *

class Config:
    def __init__(self, args):
        self.mode = args.mode
        self.data_param = args.data_param
        self.patch_size = args.patch_size
        self.is_mask = eval(args.is_mask)

        self.is_train = eval(args.is_train)
        self.is_patch_extraction = eval(args.is_patch_extraction)
        self.is_test = eval(args.is_test)
        self.is_analysis = eval(args.is_analysis)

        self.encoder = args.encoder

        self.encoder_layers_per_block = args.encoder_layers_per_block
        self.embedding_dim = args.embedding_dim
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.kl_weight = args.kl_weight
        self.info = args.info
        self.tumor_type = args.tumor_type

        ## Set experiment name
        self.get_experiment_name()

        # set device
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')

        # Early stopping parameters
        self.early_stopping_min_delta = 0.0001
        self.early_stopping_patience = 20

        # Reduce learning rate on plateau parameters
        self.learning_rate_decay = 0.5
        self.min_learning_rate = 0.000001
        self.reduce_lr_patience = 5

        self.tumor_types = ['benign', 'pathologicalbenign', 'udh', 'adh', 'fea', 'dcis', 'malignant']
        self.tumor_labels = [0, 1, 1, 2, 2, 3, 4]

        if self.data_param == 'local':
            self.base_path = '/Users/pus/Desktop/Projects/Data/Histocartography/'
            self.base_img_dir = self.base_path + 'PASCALE/Images_norm/'
            self.base_nuclei_dir = self.base_path + 'PASCALE_NEW/nuclei_info/'
            self.explanation_path = self.base_path + 'PASCALE/explanation/5_classes/'  # train, test, val

        elif self.data_param == 'dataT':
            self.base_path = '/dataT/pus/histocartography/Data/'

            #self.base_img_dir = self.base_path + 'BRACS_S/Images_norm/'
            self.base_img_dir = self.base_path + 'BRACS_L/Images_norm/'
            #self.base_img_dir = self.base_path + 'BACH/train/Images_norm/'
            #self.base_img_dir = self.base_path + 'BACH/test/Images_norm/'

            #self.base_nuclei_dir = self.base_path + 'BRACS_S/nuclei_info/'
            self.base_nuclei_dir = self.base_path + 'BRACS_L/nuclei_info/'
            #self.base_nuclei_dir = self.base_path + 'BACH/train/nuclei_info/'
            #self.base_nuclei_dir = self.base_path + 'BACH/test/nuclei_info/'

            self.explanation_path = '/dataT/gja/histocartography/data/explanations/5_classes/'

        ## Create directories
        create_directory(self.base_nuclei_dir)

        self.nuclei_detected_path = self.base_nuclei_dir + 'nuclei_detected/'
        create_directory(self.nuclei_detected_path)
        create_directory(self.nuclei_detected_path + 'instance_map/')
        create_directory(self.nuclei_detected_path + 'centroids/')
        self.create_tumor_wise_folders(self.nuclei_detected_path + 'instance_map/')
        self.create_tumor_wise_folders(self.nuclei_detected_path + 'centroids/')

        self.nuclei_features_path = self.base_nuclei_dir + 'nuclei_features/'
        create_directory(self.nuclei_features_path)
        self.nuclei_features_path += self.experiment_name + '/'
        create_directory(self.nuclei_features_path)
        self.create_tumor_wise_folders(self.nuclei_features_path)

        if 'vae' in self.mode:
            self.vae_data_path = self.base_nuclei_dir + 'vae_data/'
            create_directory(self.vae_data_path)

            self.vae_patches_path = self.vae_data_path + 'patches/'
            create_directory(self.vae_patches_path)
            create_directory(self.vae_patches_path + 'train/')
            create_directory(self.vae_patches_path + 'val/')
            create_directory(self.vae_patches_path + 'test/')

            self.vae_masks_path = self.vae_data_path + 'masks/'
            create_directory(self.vae_masks_path)
            create_directory(self.vae_masks_path + 'train/')
            create_directory(self.vae_masks_path + 'val/')
            create_directory(self.vae_masks_path + 'test/')

            self.model_save_path = self.base_nuclei_dir + 'vae_model/'
            create_directory(self.model_save_path)
            self.model_save_path += self.experiment_name + '/'
            create_directory(self.model_save_path)
        #endif
    #enddef

    def create_tumor_wise_folders(self, path):
        for t in self.tumor_types:
            create_directory(path + t)
    #enddef

    def get_experiment_name(self):
        if self.mode == 'features_hc':
            self.experiment_name = 'features_hc' + '_' + self.info

        elif self.mode == 'features_cnn':
            self.experiment_name = 'features_cnn_' + self.encoder + '_mask_' + str(self.is_mask) + '_' + self.info

        elif self.mode == 'features_vae':
            if self.encoder == 'None':
                self.experiment_name = 'features_vae_m' + str(self.encoder_layers_per_block) + '_e' + str(self.embedding_dim) + \
                                       '_bs' + str(self.batch_size) + '_lr' + str(self.learning_rate).split('.')[1] + '_mask_' + str(self.is_mask) + '_' + self.info

            else:
                self.experiment_name = 'features_vae_encoder_' + self.encoder + '_bs' + str(self.batch_size) + \
                                       '_lr' + str(self.learning_rate).split('.')[1] + '_mask_' + str(self.is_mask) + '_' + self.info
    #enddef
#end



