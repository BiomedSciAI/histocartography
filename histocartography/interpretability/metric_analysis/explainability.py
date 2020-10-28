import glob
from explanation import *
from distance import *
from plotting import *
import numpy as np
import random
import math
from scipy.stats import wasserstein_distance

class Explainability:
    def __init__(self, args, config, explainer, percentage, verbose=False, visualize=False):
        self.args = args
        self.config = config
        self.explainer = explainer
        self.percentage = percentage
        self.verbose = verbose
        self.visualize = visualize
        self.nuclei_selection_type = args.nuclei_selection_type
        self.rm_non_epithelial_nuclei = args.rm_non_epithelial_nuclei
        self.rm_misclassification = args.rm_misclassification

        self.explainer_path = config.explainer_path + str(args.classification_mode) + '/' + explainer + '/'
        self.n_tumors = len(np.unique(self.config.tumor_labels))

        # Only to get all the node information. Random indexing is applied at the end
        if self.explainer == 'Random':
            self.explainer_path = config.explainer_path + str(args.classification_mode) + '/GNNExplainer/'
            self.nuclei_selection_type = 'random'


    def get_node_info(self, exp):
        node_importance = exp.node_importance
        node_label = exp.node_label
        node_concept = exp.node_concept
        node_centroid = exp.node_centroid
        is_correct = exp.label_index == np.argmax(exp.logits)

        # Select epithelial nuclei
        if self.rm_non_epithelial_nuclei:
            idx = np.sort(np.where((node_label==0) | (node_label==1) | (node_label==2))[0])
            node_importance = node_importance[idx]
            node_label = node_label[idx]
            node_concept = node_concept[idx]
            node_centroid = node_centroid[idx]

        return node_importance, node_label, node_concept, node_centroid, is_correct


    def get_sample_explanation(self, path):
        exp = Explanation(path, self.args, self.config)

        # extract image name 
        image_name = path.split('/')[-1].split('.')[0].replace('_explanation', '')

        # Get all epithelial nuclei information
        node_importance, node_label, node_concept, node_centroid, is_correct = self.get_node_info(exp)
        return node_importance, node_label, node_concept, node_centroid, is_correct, image_name


    def get_tumor_explanation(self, tumor_type):
        paths = glob.glob(self.explainer_path + tumor_type + '/*.json')
        node_importance = []
        node_concept = []
        node_label = []
        node_centroid = []
        image_names = []

        for i in range(len(paths)):
            basename = os.path.basename(paths[i]).split('.')[0].replace('_explanation', '')

            if basename not in self.config.samples:
                continue

            importance, label, concept, centroid, is_correct, image_name = self.get_sample_explanation(paths[i])

            if len(importance) != 0:
                # if remove misclassication is true and sample is wrongly predicted, we don't append it 
                if not self.rm_misclassification or is_correct:
                    node_importance.append(importance)
                    node_concept.append(concept)
                    node_label.append(label)
                    node_centroid.append(centroid)
                    image_names.append(image_name)

        return node_importance, node_label, node_concept, node_centroid, image_names


    def get_explanation(self):
        self.node_importance = []
        self.node_concept = []
        self.node_label = []
        self.node_centroid = []
        self.image_names = []

        for t in self.config.tumor_types:
            importance, label, concept, centroid, names = self.get_tumor_explanation(tumor_type=t)

            self.node_importance.append(importance)        # list[list[array]]
            self.node_label.append(label)                  # list[list[array]]
            self.node_concept.append(concept)              # list[list[array]]
            self.node_centroid.append(centroid)            # list[list[array]]
            self.image_names.append(names)

        # Outlier removal from node concept & node importance
        # self.outlier_removal()

        # Normalize the node concepts across all samples and all tumor types
        self.normalize_node_concept()

        # Normalize the node importances per sample
        self.normalize_node_importance()

        # Get explanation per 'percentage': nuclei selection
        self.select_node_info()

        self.samples = np.array([])
        self.labels = np.array([])
        for x in self.node_label:
            self.samples = np.append(self.samples, len(x))
            for y in x:
                self.labels = np.append(self.labels, y)

        if self.verbose:
            self.printing()


    def merge_concepts_per_tumor_type(self, input):
        output = []

        for i in range(self.n_tumors):
            idx = np.where(self.config.tumor_labels == i)[0]
            output_ = []

            for j in range(len(idx)):
                output_ += input[idx[j]]

            for j in range(len(output_)):
                if j == 0:
                    output__ = output_[j]
                else:
                    output__ = np.vstack((output__, output_[j]))
            output.append(output__)
        return output


    def normalize_node_importance(self):
        for i in range(len(self.node_importance)):
            for j in range(len(self.node_importance[i])):
                self.node_importance[i][j] = normalize(self.node_importance[i][j])


    def normalize_node_concept(self):
        for i in range(len(self.node_concept)):
            for j in range(len(self.node_concept[i])):
                if i == 0 and j == 0:
                    concept = self.node_concept[i][j]
                else:
                    concept = np.append(concept, self.node_concept[i][j], axis=0)

        minm = np.min(concept, axis=0)
        maxm = np.max(concept, axis=0)
        for k in range(maxm.size):
            if maxm[k] - minm[k] != 0:
                for i in range(len(self.node_concept)):
                    for j in range(len(self.node_concept[i])):
                        self.node_concept[i][j][:, k] = (self.node_concept[i][j][:, k] - minm[k])/ (maxm[k] - minm[k])


    def select_node_info(self):
        for i in range(len(self.node_importance)):
            for j in range(len(self.node_importance[i])):
                if self.nuclei_selection_type == 'thresh':
                    # nuclei selection based on hard threshold p
                    idx = np.where(self.node_importance[i][j] > (1 - self.percentage))[0]

                elif self.nuclei_selection_type == 'cumul':
                    # nuclei selection based on cumulutative node importance p
                    idx = self.get_cumulative_pruned_index(self.node_importance[i][j], self.percentage)

                elif self.nuclei_selection_type == 'absolute':
                    idx = (-self.node_importance[i][j]).argsort()[:int(self.percentage)]

                elif self.nuclei_selection_type == 'random':
                    random.seed(0)
                    idx = list(range(len(self.node_importance[i][j])))
                    # idx = random.sample(range(len(self.node_importance[i][j])), min(len(self.node_importance[i][j]), 200))
                else:
                    raise ValueError('Unsupported nuclei selection strategy. Current options are: "thresh", "cumul", "absolute" and "random".')

                self.node_importance[i][j] = self.node_importance[i][j][idx]
                self.node_concept[i][j] = self.node_concept[i][j][idx]
                self.node_label[i][j] = self.node_label[i][j][idx]
                self.node_centroid[i][j] = self.node_centroid[i][j][idx]


    def get_cumulative_pruned_index(self, node_importance, p):
        total_node_importance = np.sum(node_importance)
        keep_node_importance = total_node_importance * p

        indices_node_importance = np.flip(np.argsort(node_importance))
        sorted_node_importance = node_importance[indices_node_importance]

        node_idx_to_keep = []
        culumative_node_importance = 0
        for node_imp, idx in zip(sorted_node_importance, indices_node_importance):
            culumative_node_importance += node_imp
            if culumative_node_importance <= keep_node_importance + 10e-3:
                node_idx_to_keep.append(idx.item())
            else:
                break

        if len(node_idx_to_keep) == 0:
            node_idx_to_keep.append(indices_node_importance[0])

        node_idx_to_keep = sorted(node_idx_to_keep)
        return node_idx_to_keep


    def outlier_removal(self):
        concept = np.array([])
        for x in self.node_concept:
            for y in x:
                concept = np.append(concept, y)

        # Detect threshold
        p = 1
        while np.sum(concept <= np.max(concept) * p)/concept.size > 0.99:
            p = p - 0.1
        threshold = np.max(concept) * (p + 0.1)

        # Outlier removal
        for i in range(len(self.node_concept)):
            for j in range(len(self.node_concept[i])):
                idx = np.where(self.node_concept[i][j] > threshold)[0]
                self.node_concept[i][j] = np.delete(self.node_concept[i][j], idx, axis=0)
                self.node_importance[i][j] = np.delete(self.node_importance[i][j], idx, axis=0)


    def printing(self):
        print('\nNode label distribution:')
        for x in self.node_label:
            labels = np.array([])
            for y in x:
                labels = np.append(labels, y)
            _, count = np.unique(labels, return_counts=True)

            print('#TRoI: ', len(x), ' #Nodes: ', len(labels), ' %Label: ', np.round(count/len(labels), 2))

