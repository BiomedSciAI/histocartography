import os
import sys
import torch

from histocartography.dataloader.constants import get_tumor_type_to_label, get_number_of_classes, get_label_to_tumor_type


class Config:
    def __init__(self, args):
        self.mode = args.mode
        self.data_param = args.data_param
        self.split = int(args.split)
        self.is_extraction = eval(args.is_extraction)
        self.model_type = args.model_type
        self.is_pretrained = eval(args.is_pretrained)

        self.patch_size = args.patch_size
        self.patch_scale = args.patch_scale
        self.num_epochs = int(args.num_epochs)
        self.batch_size = int(args.batch_size)
        self.loss = args.loss
        self.optimizer = args.optimizer
        self.learning_rate = float(args.learning_rate)
        self.dropout = float(args.dropout)
        self.weight_merge = eval(args.weight_merge)
        self.num_features = -1
        self.in_ram = args.in_ram
        self.class_split = args.class_split

        # set device
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')

        # Early stopping parameters
        self.early_stopping_min_delta = 0.001
        self.early_stopping_patience = 10

        # Reduce learning rate on plateau parameters
        self.learning_rate_decay = 0.5
        self.min_learning_rate = 0.000001
        self.reduce_lr_patience = 4

        sys.path.append('./model_zoo/')
        sys.path.append('../evaluation/')

        if self.data_param == 'local':
            self.base_path = '/Users/pus/Desktop/Projects/Data/Histocartography/BRACS_L/'
            self.base_patches_path = '/Users/pus/Desktop/Projects/Data/Histocartography/BRACS_L/baseline_patches'
        elif self.data_param == 'dataT':
            self.base_path = '/dataT/pus/histocartography/Data/BRACS_L/'
            self.base_patches_path = os.path.join(self.base_path, 'baseline_patches')

        self.base_img_path = self.base_path + 'Images_norm/'
        self.base_data_split_path = self.base_path + \
            'data_split_cv/data_split_' + str(self.split) + '/'
        self.base_model_save_path = self.base_path + 'models/'
        self.tumor_type_to_label = get_tumor_type_to_label(self.class_split)
        self.num_classes = get_number_of_classes(self.class_split)
        self.experiment_name = 'resnet34_' + '_ps' + self.patch_size + '_class_split_' + self.class_split + '_pt'

        self.create_directories()
    # enddef

    def create_directory(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
    # enddef

    def create_directories(self):
        self.create_directory(self.base_patches_path)
        self.create_directory(self.base_model_save_path)

        self.base_model_save_path += 'baseline/'
        self.create_directory(self.base_model_save_path)
        self.create_directory(
            self.base_model_save_path +
            self.model_type +
            '/')
        self.create_directory(
            self.base_model_save_path +
            self.model_type +
            '/' +
            self.experiment_name +
            '/')
        self.create_directory(self.base_model_save_path +
                              self.model_type +
                              '/' +
                              self.experiment_name +
                              '/split_' +
                              str(self.split) +
                              '/')

        self.model_save_path = self.base_model_save_path + self.model_type + \
            '/' + self.experiment_name + '/split_' + str(self.split) + '/'
    # enddef
