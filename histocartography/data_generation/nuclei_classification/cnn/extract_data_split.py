import glob
import copy
from utils import *

class DataSplit:
    def __init__(self, config):
        self.config = config
        self.tumor_types = self.config.tumor_types
        self.base_annotation_info_path = self.config.base_annotation_info_path
        self.base_data_split_path = self.config.base_data_split_path
        self.base_patches_path = self.config.base_patches_path
        self.nuclei_types = self.get_nuclei_types()


    def prepare_data_split(self):
        train_list = []
        val_list = []
        test_list = []

        for t in self.tumor_types:
            annotation_info_paths = glob.glob(self.base_annotation_info_path + t + '/*.h5')
            annotation_info_paths = [os.path.basename(x).split('.')[0] for x in annotation_info_paths]
            annotation_info_paths.sort()

            np.random.seed(0)
            test_idx = np.random.choice(np.arange(len(annotation_info_paths)),
                                        size=int(0.15 * len(annotation_info_paths)),
                                        replace=False)
            train_val_idx = np.delete(np.arange(len(annotation_info_paths)), test_idx, axis=0)

            np.random.seed(1)
            pseudo_val_idx = np.random.choice(np.arange(len(train_val_idx)),
                                              size=int(0.15 * len(train_val_idx)),
                                              replace=False)
            pseudo_train_idx = np.delete(np.arange(len(train_val_idx)), pseudo_val_idx, axis=0)

            val_idx = train_val_idx[pseudo_val_idx]
            train_idx = train_val_idx[pseudo_train_idx]

            train_list += [annotation_info_paths[x] for x in train_idx]
            val_list += [annotation_info_paths[x] for x in val_idx]
            test_list += [annotation_info_paths[x] for x in test_idx]

        print('#Train samples: ', len(train_list))
        print('#Val samples: ', len(val_list))
        print('#Test samples: ', len(test_list))

        create_directory(self.base_data_split_path)
        self.save_to_txt(train_list, self.base_data_split_path + 'train.txt')
        self.save_to_txt(val_list, self.base_data_split_path + 'val.txt')
        self.save_to_txt(test_list, self.base_data_split_path + 'test.txt')

        print('Nuclei types: ', self.nuclei_types, '\n')
        train_count = self.count_nuclei(train_list, 'Train')
        val_count = self.count_nuclei(val_list, 'Val')
        test_count = self.count_nuclei(test_list, 'Test')

        print('Total nuclei= ', np.sum(np.array([np.sum(train_count), np.sum(val_count), np.sum(test_count)])))


    def get_nuclei_types(self):
        nuclei_types = copy.deepcopy(self.config.nuclei_types)
        nuclei_types = [x.lower() for x in nuclei_types]

        # Remove nuclei type = 'NA'
        idx = nuclei_types.index('na')
        del nuclei_types[-idx]
        return nuclei_types


    def save_to_txt(self, savelist, savepath):
        with open(savepath, 'w') as output:
            for row in savelist:
                output.write(row + '\n')


    def count_nuclei(self, savelist, text=''):
        print('*********************************************************************', text)
        nuclei_count = np.zeros(len(self.nuclei_types))

        for i in savelist:
            for t in range(len(self.nuclei_types)):
                path = self.base_patches_path + self.nuclei_types[t] + '/' + i + '_*.png'
                nuclei_count[t] += len(glob.glob(path))

        print(nuclei_count, ':', np.sum(nuclei_count))
        return nuclei_count