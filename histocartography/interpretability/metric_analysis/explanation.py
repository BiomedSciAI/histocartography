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
        self.node_idx_to_keep = np.arange(len(self.node_importance))

    def read_concept(self):
        with h5py.File(self.config.features_path + self.tumor_type + '/' + self.basename + '.h5', 'r') as f:
            self.embeddings = np.array(f['embeddings'])

        idx = self.config.feature_names.index(self.args.concept)
        self.node_concept = self.embeddings[self.node_idx_to_keep, idx]
