import numpy as np
from distance import Distance
from sklearn.metrics import precision_score


TUMOR_LABEL_TO_RELEVANT_NUCLEI_TYPE = {
    0: 0,
    1: 1,
    2: 2
}

class Metric:
    def __init__(self, args, config, explainer, percentage, explanation, verbose=False):
        self.args = args
        self.config = config
        self.explainer = explainer
        self.percentage = percentage
        self.explanation = explanation
        self.verbose = verbose
        self.n_tumors = len(np.unique(config.tumor_labels))

        # Merge correlation values per tumor group
        self.concept = self.merge_per_tumor_type(self.explanation.node_concept)
        self.nuclei_labels = self.merge_per_tumor_type(self.explanation.node_label)

        # Get similarity function
        self.dist = Distance(self.args.similarity)

    def merge_per_tumor_type(self, input):
        output = [np.array([]) for i in range(self.n_tumors)]

        for i in range(self.n_tumors):
            idx = np.where(self.config.tumor_labels == i)[0]
            for id in idx:
                if isinstance(input[id], float):
                    output[i] = np.append(output[i], input[id])
                else:
                    for x in input[id]:
                        output[i] = np.append(output[i], x)

        return output

    def compute_tumor_similarity(self):
        M = np.zeros(shape=(self.n_tumors, self.n_tumors))

        # Tumor similarity
        for i in range(self.n_tumors):
            x = np.reshape(self.concept[i], newshape=(-1, 1))

            for j in range(self.n_tumors):
                y  = np.reshape(self.concept[j], newshape=(-1, 1))
                M[i, j] = self.dist.similarity(x, y, metric='l1')
                M[j, i] = self.dist.similarity(y, x, metric='l1')

        self.tumor_similarity = np.round(M, 4)

        if self.verbose:
            print('**********************************************')
            print('Tumor similarity: ' + self.explainer + ' : p= ', self.percentage, ':')
            print(self.tumor_similarity)
            print()


    def get_risk(self):
        self.risk = np.ones(shape=(self.n_tumors, self.n_tumors))
        if eval(self.args.risk):
            for i in range(self.n_tumors):
                for j in range(self.n_tumors):
                    self.risk[i, j] = abs(i - j)


    def compute_score(self):
        self.compute_tumor_similarity()
        self.get_risk()
        self.score = np.multiply(self.tumor_similarity, self.risk)
        return round(np.sum(self.score) / 2, 4)


    def compute_nuclei_selection_relevance(self):
        all_precisions = []
        print('Nuclei types: ', self.config.nuclei_types[1:], '\n')

        for tumor_type, nuclei_per_tumor_type in enumerate(self.nuclei_labels):
            print('Tumor type: ', tumor_type)

            precision_tumor = sum(nuclei_per_tumor_type == tumor_type) / len(nuclei_per_tumor_type)
            precision_epi = sum((nuclei_per_tumor_type == 0) + (nuclei_per_tumor_type == 1) + (nuclei_per_tumor_type == 2) + (nuclei_per_tumor_type == 5)) / len(nuclei_per_tumor_type)
            print('Precision for tumor:', round(precision_tumor, 4))
            print('Precision for selecting epi:', round(precision_epi, 4))
            all_precisions.append(precision_epi)

            nuclei, count = np.unique(np.asarray(nuclei_per_tumor_type), return_counts=True)
            count = np.round(count/np.sum(count), 4)
            print('Nuclei classes distribution: ',  count, '\n')

        return round(sum(all_precisions) / len(all_precisions), 4)












