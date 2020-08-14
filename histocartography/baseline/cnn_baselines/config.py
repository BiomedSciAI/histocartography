import torch
import numpy as np
from utils import *

class Config:
    def __init__(self, args):
        self.mode = args.mode
        self.encoder = args.encoder
        self.aggregator = args.aggregator

        self.class_split = args.class_split
        self.cv_split = args.cv_split
        self.is_train = eval(args.is_train)

        self.is_finetune = eval(args.is_finetune)
        self.patch_size = args.patch_size       #patch size at 40x
        self.patch_scale = args.patch_scale

        self.info = args.info

        # Set device
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')

        # Get magnification
        self.get_magnifications()

        # Get classes
        self.get_tumor_classes()

        # Set training parameters
        self.set_training_parameters(args)

        ## Create directories
        self.set_paths()


    def set_paths(self):
        self.base_path = '/dataT/pus/histocartography/Data/'
        self.base_img_path = self.base_path + 'BRACS_L/Images_norm/'
        self.base_patches_path = '/dataL/pus/histocartography/Data/BRACS_L/baseline_patches/'
        self.base_model_save_path = self.base_path + 'BRACS_L/cnn_models/'

        self.base_data_split_path = self.base_path + 'BRACS_L/data_split_cv_correct/data_split_' + str(self.cv_split) + '/'
        self.create_directories()


    def get_magnifications(self):
        if self.mode == 'single_scale_10x':
            self.magnifications = ['10x']
        elif self.mode == 'single_scale_20x':
            self.magnifications = ['20x']
        elif self.mode == 'single_scale_40x':
            self.magnifications = ['40x']
        elif self.mode == 'late_fusion_1020x':
            self.magnifications = ['10x', '20x']
        else:
            self.magnifications = ['10x', '20x', '40x']


    def get_tumor_classes(self):
        tumor_types = self.class_split.replace('VS', '+').split('+')
        class_types = self.class_split.split('VS')

        tumor_labels = []
        ctr = 0
        for c in class_types:
            c = c.split('+')
            for t in tumor_types:
                if t in c:
                    tumor_labels.append(ctr)
            ctr += 1

        self.tumor_types = tumor_types
        self.tumor_labels = tumor_labels
        self.num_classes = len(np.unique(self.tumor_labels))


    def set_training_parameters(self, args):
        # Early stopping parameters
        self.early_stopping_min_delta = 0.0001
        self.early_stopping_patience = 10

        # Reduce learning rate on plateau parameters
        self.learning_rate_decay = 0.5
        self.min_learning_rate = 0.000001
        self.reduce_lr_patience = 3

        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.dropout = args.dropout


    def create_directories(self):
        create_directory(self.base_patches_path)
        create_directory(self.base_model_save_path)

        if self.mode != 'extract_patches':
            self.model_save_path = self.base_model_save_path + \
                                   self.encoder + '_' + self.mode + '_ps' + str(self.patch_size) + \
                                   '_bs' + str(self.batch_size) + '_lr' + str(self.learning_rate) + '_' + self.info + '/'
            create_directory(self.model_save_path)

            if self.class_split == 'benignVSpathologicalbenignVSudhVSadhVSfeaVSdcisVSmalignant':
                class_split = 'C7'
            elif self.class_split == 'benignVSpathologicalbenign+udhVSadh+feaVSdcis+malignant':
                class_split = 'C4'
            elif self.class_split == 'benign+pathologicalbenign+udh+adh+fea+dcisVSmalignant':
                class_split = 'MvsALL'
            elif self.class_split == 'benign+pathologicalbenign+udhVSadh+fea+dcis':
                class_split = 'BPUvsAFD'
            elif self.class_split == 'benignVSpathologicalbenign+udh':
                class_split = 'BvsPU'
            elif self.class_split == 'pathologicalbenignVSudh':
                class_split = 'PvsU'
            elif self.class_split == 'adh+feaVSdcis':
                class_split = 'AFvsD'
            elif self.class_split == 'adhVSfea':
                class_split = 'AvsF'
            else:
                print('ERROR: Invalid class split')
                exit()

            self.model_save_path += class_split + '/'
            create_directory(self.model_save_path)

            if self.cv_split != -1:
                self.model_save_path += str(self.cv_split) + '/'
                create_directory(self.model_save_path)




