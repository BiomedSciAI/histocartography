import json
import h5py
import numpy as np
import os

class Explanation:
    def __init__(self, path, args, config):
        self.args = args
        self.config  = config
        self.path = path
        self.basename = os.path.basename(self.path).split('.')[0].replace('_explanation', '')
        self.tumor_type = self.basename.split('_')[1]

        self.read_json()
        self.read_concept()

    def read_json(self):
        with open(self.path) as f:
            data = json.load(f)
            output = data['output']

        self.label_index = output['label_index']
        explanation = output['explanation']['1']        # str(self.args.p)
        self.logits = np.asarray(explanation['logits'])
        self.num_nodes = explanation['num_nodes']
        self.num_edges = explanation['num_edges']

        self.node_importance = np.asarray(explanation['node_importance'])
        self.node_centroid = np.asarray(explanation['centroid'])
        self.node_label = np.asarray(explanation['nuclei_label'])

    def read_concept(self):
        if self.args.concept == ['type']:

            self.node_label = self.node_label.astype(int) 
            hack_node_labels = False
            if hack_node_labels:
                if self.label_index == 0:  # image is benign
                    node_labels_filtered = np.where(self.node_label == 1, 0, self.node_label)   # replace atypical by benign  
                    node_labels_filtered = np.where(node_labels_filtered == 2, 0, node_labels_filtered) # replace tumor by benign  
                elif self.label_index == 1:  # image is atypical 
                    node_labels_filtered = np.where(self.node_label == 0, 1, self.node_label)   # replace benign by atypical  
                    node_labels_filtered = np.where(node_labels_filtered == 2, 1, node_labels_filtered) # replace tumor by atypical 
                elif self.label_index == 2:  # image is invasive 
                    node_labels_filtered = np.where(self.node_label == 0, 2, self.node_label)   # replace benign by invasive  
                    node_labels_filtered = np.where(node_labels_filtered == 1, 2, node_labels_filtered) # replace atypical by invasive
                else:
                    print('Weird image label', self.label_index)
            else:
                node_labels_filtered = self.node_label

            # if the concept is the nuclei type --> one hot encoding 
            num_nuclei_types = 6
            self.node_concept = np.zeros((node_labels_filtered.size, num_nuclei_types)).astype(int)
            self.node_concept[np.arange(node_labels_filtered.size), node_labels_filtered] = 1
        else:
            with h5py.File(self.config.features_path + self.tumor_type + '/' + self.basename + '.h5', 'r') as f:
                self.embeddings = np.array(f['embeddings'])
            idx = []
            for x in self.args.concept:
                idx.append(self.config.feature_names.index(x))
            idx = [int(x) for x in idx]
            self.node_concept = self.embeddings[:, idx]
