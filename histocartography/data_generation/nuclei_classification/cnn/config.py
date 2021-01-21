from utils import *
import torch

class Config:
    def __init__(self, args):
        self.set_paths(args)

        # Nuclei info
        self.patch_size = 32
        self.nuclei_types = ['NA', 'Normal', 'Atypical', 'Tumor', 'Stromal', 'Lymphocyte', 'Dead']    # nuclei names as per qupath annotation
        self.nuclei_labels = [-1, 0, 1, 2, 3, 4, 5]
        self.nuclei_colors = ['k', 'b', '#4d001a', 'm', 'lime', 'c', 'y']

        # Tumor info
        self.tumor_types = ['benign', 'pathologicalbenign', 'udh', 'adh', 'fea', 'dcis', 'malignant']

        # Set device
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')


    def set_paths(self, args):
        if args.data_param == 'local':
            base_path = '/Users/pus/Desktop/Projects/Data/Histocartography/Nuclei/'

            self.base_annotation_path = base_path + 'annotation/'
            self.base_img_path = self.base_annotation_path + 'images/'
            self.base_centroid_path = self.base_annotation_path + 'centroids/'
            self.base_instance_map_path = self.base_annotation_path + 'instance_map/'
            self.base_masks_path = self.base_annotation_path + 'annotation_masks/'
            self.base_annotation_info_path = self.base_annotation_path + 'annotation_info/'
            self.base_overlaid_path = self.base_annotation_path + 'annotation_overlaid/'

            self.base_train_path = base_path + 'train/'
            self.base_data_split_path = self.base_train_path + 'data_split/'
            self.base_patches_path = self.base_train_path + 'patches/'
            self.base_model_save_path = self.base_train_path + 'models/'
            self.pretrained_model_path = self.base_train_path + 'pretrained_models/'

            self.base_predict_path = base_path + 'prediction/'
            self.base_predict_img_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/Images_norm/'
            self.base_predict_samples_path = self.base_predict_path + 'prediction_samples/'
            self.base_predict_info_path = self.base_predict_path + 'prediction_info/'
            self.base_predict_overlaid_path = self.base_predict_path + 'prediction_overlaid/'

        elif args.data_param == 'dataT':
            base_path = '/dataT/pus/histocartography/Data/Nuclei/'

            self.base_annotation_path = base_path + 'annotation/'
            self.base_img_path = self.base_annotation_path + 'images/'
            self.base_centroid_path = '/dataT/pus/histocartography/Data/BRACS_L/nuclei_info/nuclei_detected/centroids/'
            self.base_instance_map_path = '/dataT/pus/histocartography/Data/BRACS_L/nuclei_info/nuclei_detected/instance_map/'
            self.base_masks_path = self.base_annotation_path + 'annotation_masks/'
            self.base_annotation_info_path = self.base_annotation_path + 'annotation_info/'
            self.base_overlaid_path = self.base_annotation_path + 'annotation_overlaid/'

            self.base_train_path = base_path + 'train/'
            self.base_data_split_path = self.base_train_path + 'data_split/'
            self.base_patches_path = self.base_train_path + 'patches/'
            self.base_model_save_path = self.base_train_path + 'models/'
            self.pretrained_model_path = self.base_train_path + 'pretrained_models/'

            self.base_predict_path = base_path + 'prediction/'
            self.base_predict_img_path = '/dataT/pus/histocartography/Data/BRACS_L/Images_norm/'
            self.base_predict_samples_path = self.base_predict_path + 'prediction_samples/'
            self.base_predict_info_path = self.base_predict_path + 'prediction_info/'
            self.base_predict_overlaid_path = self.base_predict_path + 'prediction_overlaid/'

        self.model_save_path = self.base_model_save_path + args.arch + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + \
                               '_pt_' + args.pretrained + '_ft_' + args.finetune + '_wl_' + args.weighted_loss + '/'

        create_directory(self.base_annotation_path)
        create_directory(self.base_train_path)
        create_directory(self.base_predict_path)

        if args.mode == 'extract_annotations':
            create_directory(self.base_masks_path)
            create_directory(self.base_overlaid_path)
            create_directory(self.base_annotation_info_path)

        elif args.mode == 'extract_patches':
            create_directory(self.base_patches_path)

        elif args.mode == 'extract_data_split':
            create_directory(self.base_data_split_path)

        elif args.mode == 'train':
            create_directory(self.base_model_save_path)
            create_directory(self.model_save_path)

        elif args.mode == 'predict':
            create_directory(self.base_predict_info_path)
            create_directory(self.base_predict_overlaid_path)














