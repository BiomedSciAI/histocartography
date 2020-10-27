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

        # Merge values per tumor group
        self.concept = self.merge_concepts_per_tumor_type(self.explanation.node_concept)
        self.nuclei_labels = self.merge_labels_per_tumor_type(self.explanation.node_label)

        # Get distance function
        self.dist = Distance(self.args.distance)


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


    def merge_labels_per_tumor_type(self, input):
        output = []
        for i in range(self.n_tumors):
            idx = np.where(self.config.tumor_labels == i)[0]
            output_ = []
            for id in idx:
                input[id] = [np.asarray(x) for x in input[id]]
                output_ += input[id]
            output.append(output_)
        return output


    def get_distance(self, input):
        M = np.zeros(shape=(self.n_tumors, self.n_tumors))

        # Tumor distance
        for i in range(len(input)):
            for j in range(len(input)):
                M[i, j] = self.dist.distance(input[i], input[j], metric='euclidean')
                M[j, i] = self.dist.distance(input[j], input[i], metric='euclidean')
        return np.round(M, 4)


    def get_risk(self):
        risk = np.ones(shape=(self.n_tumors, self.n_tumors))
        if eval(self.args.risk):
            for i in range(self.n_tumors):
                for j in range(self.n_tumors):
                    risk[i, j] = abs(i - j)
        return risk


    def compute_concept_score(self):
        distance =  self.get_distance(self.concept)
        risk = self.get_risk()
        score = np.sum(np.multiply(distance, risk)) / 2

        #print('********** Concept-based tumor distance')
        #print(distance, '\n')

        return round(score, 4)


    def compute_nuclei_score(self):
        # Score based on per sample nuclei statistics
        nuclei = []
        for i in range(len(self.nuclei_labels)):
            for j in range(len(self.nuclei_labels[i])):
                nuclei_ = np.zeros(len(self.config.nuclei_types[1:]))
                for k in range(nuclei_.size):
                    nuclei_[k] = sum(self.nuclei_labels[i][j] == k)
                    '''
                    mask = self.nuclei_labels[i][j] == k
                    if isinstance(self.nuclei_labels[i][j], np.float64):
                        nuclei_[k] = mask
                    else:
                        nuclei_[k] = sum(mask)
                    #'''

                if j == 0:
                    nuclei__ = nuclei_ / np.sum(nuclei_)
                else:
                    nuclei__ = np.vstack((nuclei__, nuclei_ / np.sum(nuclei_)))
            nuclei.append(nuclei__)

        distance = self.get_distance(nuclei)
        risk = self.get_risk()
        score = round(np.sum(np.multiply(distance, risk)) / 2, 4)
        #print('********** Nuclei-based tumor distance')
        #print(distance, '\n')

        # Nuclei statistics per tumor type
        precision_epi = []
        for i in range(len(self.nuclei_labels)):
            nuclei = np.array([])
            for y in self.nuclei_labels[i]:
                nuclei = np.append(nuclei, y)

            nuclei_ = np.zeros(len(self.config.nuclei_types[1:]))
            for k in range(len(nuclei_)):
                nuclei_[k] = sum(nuclei == k)

            precision_epi_ = round(sum((nuclei == 0) + (nuclei == 1) + (nuclei == 2) + (nuclei == 5)) / len(nuclei), 4)
            precision_epi.append(precision_epi_)
            #print('Tumor type: ', i, ' %Nuclei: ', np.round(nuclei_/np.sum(nuclei_), 2), ' precision_epi: ', precision_epi_)

        precision_epi = round(sum(precision_epi) / len(precision_epi), 4)
        return score, precision_epi















