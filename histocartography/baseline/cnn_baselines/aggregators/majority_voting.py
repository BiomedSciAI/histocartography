import numpy as np
import h5py
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix


class MajorityVoting:
    def __init__(self, config):
        self.model_save_path = config.model_save_path
        self.eval_mode = ['train', 'val', 'test']
        self.tumor_types = config.tumor_types
        self.tumor_labels = config.tumor_labels
        self.num_classes = config.num_classes

    def get_troi_ids(self, config, mode, tumor_type):
        trois = []
        filename = config.base_data_split_path + mode + '_list_' + tumor_type + '.txt'
        with open(filename, 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                if line != '':
                    trois.append(line)
        return trois

    def compute_statistics(self, true_labels, pred_labels):
        acc = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='weighted')

        target_names = list(np.arange(self.num_classes))
        target_names = [str(x) for x in target_names]
        cls_report = classification_report(
            true_labels, pred_labels, target_names=target_names)
        conf_matrix = confusion_matrix(true_labels, pred_labels)

        print('classification report:', cls_report)
        print('accuracy:', round(acc, 4))
        print('weighted F1:', round(f1, 4), '\n')
        print(conf_matrix)
        print('\n')

    def process(self, config):
        for mode in self.eval_mode:
            true_labels = np.array([])
            pred_labels = np.array([])

            for tumor_type in self.tumor_types:
                troi_ids = self.get_troi_ids(config, mode, tumor_type)

                true_label = self.tumor_labels[self.tumor_types.index(
                    tumor_type)]
                true_labels = np.concatenate(
                    (true_labels, np.ones(len(troi_ids)) * true_label))

                with h5py.File(self.model_save_path + mode + '_' + tumor_type + '.h5', 'r') as f:
                    patch_count = np.array(f['patch_count']).astype(int)
                    patch_probabilities = np.array(f['patch_probabilities'])

                patch_labels = np.argmax(patch_probabilities, axis=1)

                start = 0
                for i in range(len(troi_ids)):
                    labels = patch_labels[start: start + patch_count[i]]
                    start += patch_count[i]

                    count = np.arange(self.num_classes)
                    for i in range(self.num_classes):
                        count[i] = np.sum(labels == i)
                    pred_label = np.where(count == np.max(count))[0]

                    if len(pred_label) == 1:
                        pred_labels = np.append(pred_labels, pred_label)
                    else:
                        if true_label in pred_label:
                            pred_labels = np.append(pred_labels, true_label)
                        else:
                            pred_labels = np.append(pred_labels, pred_label[0])

            print(
                'STATISTICS for ',
                mode,
                ' *************************************************************************')
            self.compute_statistics(true_labels, pred_labels)
