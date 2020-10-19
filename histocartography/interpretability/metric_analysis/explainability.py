import glob
from explanation import *
from utils import *
from distance import *
from scipy import stats
from plotting import *
from matplotlib import pyplot as plt
from skimage import exposure
from PIL import ImageDraw, Image

class Explainability:
    def __init__(self, args, config, explainer, percentage, verbose=False, visualize=False):
        self.args = args
        self.config = config
        self.explainer = explainer
        self.percentage = percentage
        self.verbose = verbose
        self.visualize = visualize

        self.explainer_path = config.explainer_path + str(args.classification_mode) + '/' + explainer + '/'
        self.dist = Distance(self.args.similarity)
        self.n_tumors = len(np.unique(self.config.tumor_labels))


    def get_node_info(self, exp):
        node_importance = exp.node_importance
        node_label = exp.node_label
        node_concept = exp.node_concept
        node_centroid = exp.node_centroid

        # Select epithelial nuclei
        idx = np.sort(np.where((node_label==0) | (node_label==1) | (node_label==2))[0])
        node_importance = node_importance[idx]
        node_label = node_label[idx]
        node_concept = node_concept[idx]
        node_centroid = node_centroid[idx]

        return node_importance, node_label, node_concept, node_centroid


    def get_sample_explanation(self, path):
        exp = Explanation(path, self.args, self.config)

        # Get all epithelial nuclei information
        node_importance, node_label, node_concept, node_centroid = self.get_node_info(exp)

        return node_importance, node_label, node_concept, node_centroid


    def get_tumor_explanation(self, tumor_type):
        paths = glob.glob(self.explainer_path + tumor_type + '/*.json')

        node_importance = []
        node_concept = []
        node_label = []
        node_centroid = []

        for i in range(len(paths)):
            basename = os.path.basename(paths[i]).split('.')[0].replace('_explanation', '')

            if basename not in self.config.samples:
                continue

            importance, label, concept, centroid = self.get_sample_explanation(paths[i])

            if importance is not None:
                node_importance.append(importance)
                node_concept.append(concept)
                node_label.append(label)
                node_centroid.append(centroid)

        return node_importance, node_label, node_concept, node_centroid


    def get_explanation(self):
        self.node_importance = []
        self.node_concept = []
        self.node_label = []
        self.node_centroid = []

        for t in self.config.tumor_types:
            importance, label, concept, centroid = self.get_tumor_explanation(tumor_type=t)

            self.node_importance.append(importance)        # list[list[array]]
            self.node_label.append(label)                  # list[list[array]]
            self.node_concept.append(concept)              # list[list[array]]
            self.node_centroid.append(centroid)            # list[list[array]]

        # Outlier removal from node concept & node importance
        self.outlier_removal()

        # Normalize the node concepts across all samples and all tumor types
        self.normalize_node_concept()

        # Normalize the node importances per sample
        self.normalize_node_importance()

        # Get explanation per 'percentage'
        for i in range(len(self.node_importance)):
            for j in range(len(self.node_importance[i])):
                idx = np.where(self.node_importance[i][j] < (1 - self.percentage))[0]
                self.node_importance[i][j] = np.delete(self.node_importance[i][j], idx, axis=0)
                self.node_concept[i][j] = np.delete(self.node_concept[i][j], idx, axis=0)
                self.node_label[i][j] = np.delete(self.node_label[i][j], idx, axis=0)

        self.printing()


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


    def normalize_node_importance(self):
        for i in range(len(self.node_importance)):
            for j in range(len(self.node_importance[i])):
                self.node_importance[i][j] = normalize(self.node_importance[i][j])


    def normalize_node_concept(self):
        concept = np.array([])
        for x in self.node_concept:
            for y in x:
                concept = np.append(concept, y)

        minm = np.min(concept)
        maxm = np.max(concept)
        if maxm - minm != 0:
            for i in range(len(self.node_concept)):
                for j in range(len(self.node_concept[i])):
                    self.node_concept[i][j] = (self.node_concept[i][j] - minm)/ (maxm - minm)


    def printing(self):
        print('\nNode label distribution:')

        self.labels = np.array([])
        self.samples = np.array([])

        for x in self.node_label:
            self.samples = np.append(self.samples, len(x))

            labels = np.array([])
            for y in x:
                labels = np.append(labels, y)
            self.labels = np.append(self.labels, labels)

            if self.verbose:
                print('#TRoI: ', len(x), ' #Nodes: ', len(labels), ' Label: ', np.unique(labels, return_counts=True))
























