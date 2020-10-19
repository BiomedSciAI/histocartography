import numpy as np
from distance import Distance

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

        # Get similarity function
        self.dist = Distance(self.args.similarity)


    def merge_per_tumor_type(self, input):
        output = [np.array([]) for i in range(self.n_tumors)]

        for i in range(self.n_tumors):
            idx = np.where(self.config.tumor_labels == i)[0]
            for id in idx:
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
