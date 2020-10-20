import numpy as np
import glob
import os

class Configuration:
    def __init__(self, args):
        self.base_path = '/Users/gja/Documents/PhD/histocartography/data/explainability/'

        self.explainer_path = self.base_path + 'explainers/'
        self.features_path = self.base_path + 'nuclei_info/nuclei_features/features_hc_/'
        self.centroids_path = self.base_path + 'nuclei_info/nuclei_detected/centroids/'
        self.classes_path = self.base_path + 'nuclei_info/predictions/'
        self.figure_save_path = self.base_path + 'explainer_analysis/' + str(args.classification_mode) + '/'
        os.makedirs(self.figure_save_path, exist_ok=True)

        # Tumor types
        self.tumor_types = ['benign', 'pathologicalbenign', 'udh', 'adh', 'fea', 'dcis', 'malignant']

        # Nuclei types
        self.nuclei_types = ['NA', 'Normal', 'Atypical', 'Tumor', 'Stromal', 'Lymphocyte', 'Dead']  # nuclei names as per qupath annotation
        self.nuclei_labels = [-1, 0, 1, 2, 3, 4, 5]

        # Select tumor labels and colors
        if args.classification_mode == 2:
            self.tumor_labels = [0, 0, 0, -1, -1, 1, 1]
            self.tumor_colors = ['darkgreen', 'darkgreen', 'darkgreen', '', '', 'red', 'red']

        elif args.classification_mode == 3:
            self.tumor_labels = [0, 0, 0, 1, -1, 2, 2]  # exclude FEA samples
            self.tumor_colors = ['darkgreen', 'darkgreen', 'darkgreen', 'blue', '', 'red', 'red']

        elif args.classification_mode == 5:
            self.tumor_labels = [0, 1, 1, 2, 2, 3, 4]
            self.tumor_colors = ['lime', 'darkgreen', 'darkgreen', 'blue', 'blue', 'magenta', 'red']

        elif args.classification_mode == 7:
            self.tumor_labels = [0, 1, 2, 3, 4, 5, 6]
            self.tumor_colors = ['lime', 'darkgreen', 'teal', 'blue', 'purple', 'magenta', 'red']

        # Remove unrequired classes
        while -1 in self.tumor_labels:
            idx = self.tumor_labels.index(-1)
            del self.tumor_labels[idx]
            del self.tumor_types[idx]
            del self.tumor_colors[idx]
        self.tumor_labels = np.asarray(self.tumor_labels)

        # Set percentage
        if args.p == -1:
            self.percentages = np.arange(0.05, 1, 0.45)
        else:
            self.percentages = np.array([args.p])

        if args.explainer == '-1':
            #self.explainers = ['GraphLRP', 'GraphGradCAM', 'GraphGradCAMpp', 'GNNExplainer']
            self.explainers = ['GraphGradCAMpp', 'GNNExplainer']
        else:
            self.explainers = [args.explainer]



        # List of concepts
        self.feature_names = ['mean_fg', 'mean_diff', 'var_fg', 'skew_fg', 'mean_entropy',
                              'glcm_dissimilarity', 'glcm_homogeneity', 'glcm_energy', 'glcm_ASM',
                              'eccentricity', 'area', 'majoraxis_length', 'minoraxis_length', 'perimeter', 'solidity', 'orientation',
                              ]

        # CONCEPTS:
        # nuclei type: class prediction
        # nuclei shape: area, perimeter, majoraxis_length, minoraxis_length, circularity, eccentricity, MinorMajorAxisRatio, convexity (Solidity)
        # nuclei packing:

    def get_sample_names(self, args, explainers):
        samples = []

        for e in explainers:
            samples_ = []
            for t in self.tumor_types:
                samples__ = glob.glob(self.explainer_path +  str(args.classification_mode) + '/' + e + '/' + t + '/*.json')
                samples__ = [os.path.basename(x).split('.')[0].replace('_explanation', '') for x in samples__]
                samples_.append(samples__)

            samples.append([x for y in samples_ for x in y ])

        # Select common samples across the explainers
        samples_all = list(set([x for y in samples for x in y]))
        samples_unique = []

        for x in samples_all:
            flag = True
            for y in samples:
                if x not in y:
                    flag = False
            if flag:
                samples_unique.append(x)

        self.samples = samples_unique




















