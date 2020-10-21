from utils import *
import torch

class Config:
    def __init__(self, args):

        if args.data_param == 'local':
            base_path = '/Users/pus/Desktop/Projects/Data/Histocartography/Nuclei/'
            self.base_img_path = base_path + 'images/'
            self.base_centroid_path = base_path + 'centroids/'
            self.base_instance_map_path = base_path + 'instance_map/'
            self.base_masks_path = base_path + 'annotation_masks/'
            self.base_annotation_centroid_path = base_path + 'annotation_centroids/'
            self.base_overlaid_path = base_path + 'annotation_overlaid/'
            self.base_patches_path = base_path + 'patches/'
            self.base_model_save_path = base_path + 'models/'
            self.base_data_split_path = base_path + 'data_split/'
            self.base_test_save_path = base_path + 'predictions/'
            self.pretrained_model_path = base_path + 'pretrained/'

        elif args.data_param == 'dataT':
            base_path = '/dataT/pus/histocartography/Data/Nuclei/'
            self.base_img_path = base_path + 'images/'
            self.base_centroid_path = '/dataT/pus/histocartography/Data/BRACS_L/nuclei_info/nuclei_detected/centroids/'
            self.base_instance_map_path = '/dataT/pus/histocartography/Data/BRACS_L/nuclei_info/nuclei_detected/instance_map/'
            self.base_masks_path = base_path + 'annotation_masks/'
            self.base_annotation_centroid_path = base_path + 'annotation_centroids/'
            self.base_overlaid_path = base_path + 'annotation_overlaid/'
            self.base_patches_path = base_path + 'patches/'
            self.base_model_save_path = base_path + 'models/'
            self.base_data_split_path = base_path + 'data_split/'
            self.base_annotation_embedding_path = base_path + 'annotation_embeddings/'
            self.pretrained_model_path = base_path + 'pretrained/'

            self.base_test_img_path = '/dataT/pus/histocartography/Data/BRACS_L/Images_norm/'
            self.base_test_samples_path = base_path + 'prediction_samples/'
            self.base_test_save_path = base_path + 'predictions/'
            self.base_test_overlaid_path = base_path + 'predictions_overlaid/'


        create_directory(self.base_img_path)
        create_directory(self.base_annotation_centroid_path)
        create_directory(self.base_overlaid_path)
        create_directory(self.base_patches_path)
        create_directory(self.base_data_split_path)
        create_directory(self.base_annotation_embedding_path)
        #create_directory(self.base_test_save_path)

        self.nuclei_types = ['NA', 'Normal', 'Atypical', 'Tumor', 'Stromal', 'Lymphocyte', 'Dead']    # nuclei names as per qupath annotation
        self.nuclei_labels = [-1, 0, 1, 2, 3, 4, 5]
        self.nuclei_colors = ['k', 'b', '#4d001a', 'm', 'lime', 'c', 'y']

        self.tumor_types = ['benign', 'pathologicalbenign', 'udh', 'adh', 'fea', 'dcis', 'malignant']

        self.patch_size = 32

        # Set device
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')

        # Save model path
        create_directory(self.base_model_save_path)
        self.model_save_path = self.base_model_save_path + args.arch + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + \
                                '_pt_' + args.pretrained + '_ft_' + args.finetune + '_wl_' + args.weighted_loss + '/'
        create_directory(self.model_save_path)














