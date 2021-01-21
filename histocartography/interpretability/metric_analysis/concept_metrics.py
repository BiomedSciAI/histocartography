import numpy as np
from distance import Distance
from sklearn.metrics import precision_score
import math
import itertools
from scipy import stats
import matplotlib.pyplot as plt


class ConceptMetric:
    def __init__(self, args, config, explainer, percentage, explanation, verbose=False):
        self.args = args
        self.config = config
        self.explainer = explainer
        self.percentage = percentage
        self.explanation = explanation
        self.verbose = verbose
        self.classes = np.unique(config.tumor_labels)
        self.n_tumors = len(self.classes)

        self.concept = self.merge_concepts_per_tumor_type(self.explanation.node_concept)        # list[array] = #tumor[#nuclei x #concept]
        self.nuclei_labels = self.merge_labels_per_tumor_type(self.explanation.node_label)      # list[list[array]] = #tumor[#samples[#nuclei]]

        # Get distance function
        self.dist = Distance('wassertein')
        self.nuclei_dist = Distance('hist')

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

    def histogram_analysis(self, input, step):
        # Histogram bin edges along dimensions
        x = np.array([])
        for i in range(len(input)):
            if i == 0:
                x = input[i]
            else:
                x = np.vstack((x, input[i]))
        minm = np.min(x, axis=0)
        maxm = np.max(x, axis=0)

        bins = []
        for i in range(len(minm)):
            bins_ = np.array([])
            ctr = math.ceil((maxm[i] - minm[i]) / step)
            j = 0
            while j <= ctr:
                bins_ = np.append(bins_, minm[i] + j * step)
                j += 1
            bins.append(bins_)

        # Create one histogram for each split:
        count = []
        for split_id in range(3):  # 3 is the number of splits
            for i in range(len(input)):
                num_samples = len(input[i])
                from_ = int(num_samples / 3) * split_id
                to_ = int(num_samples / 3) * (split_id+1)
                subset = input[i][from_:to_]
                H, _ = np.histogramdd(subset, bins=bins, density=True)
                count.append(H)

        minm = np.inf
        maxm = -np.inf
        for i in range(len(count)):
            if np.min(count[i]) < minm:
                minm = np.min(count[i])
            if np.max(count[i]) > maxm:
                maxm = np.max(count[i])

        if maxm - minm != 0:
            for i in range(len(count)):
                count[i] = (count[i] - minm)/ (maxm - minm)

        return count

    def get_distance(self, input, dist):
        """
        input is organised as follow:
        - S1-C0, S1-C1, S1-C2, S2-C0, S2-C1, S2-C2, S3-C0, S3-C1, S3-C2
        """

        # 1. get all pairs of classes
        all_pairs = list(itertools.combinations(self.classes, 2))
        distance_per_pair = {}

        # 2. compute distance for each pair
        for pair in all_pairs:
            scores = []
            for split_id in range(3):
                i1 = pair[0] + split_id * 3
                i2 = pair[1] + split_id * 3
                score = dist.distance(input[i1], input[i2])
                scores.append(score)
            distance_per_pair[str(pair)] = scores

        return distance_per_pair

    def compute_tumor_type_stats(self):
        stats_per_tumor_type = {}
        for i, h in enumerate(self.concept):
            # a. compute mean
            mu = np.mean(h)
            std = np.std(h)
            ratio = std / mu
            stats_per_tumor_type[str(i)] = {
                'mean': float(np.round(mu, 4)),
                'std': float(np.round(std, 4)),
                'ratio': float(np.round(std / mu, 4))
            }
        return stats_per_tumor_type

    def compute_concept_score(self):

        # 1. compute histogram
        self.concept = self.histogram_analysis(self.concept, step=0.01)

        # 2. compute distance for each pair of classes
        distance_per_pair =  self.get_distance(self.concept, self.dist)

        return distance_per_pair

    def compute_nuclei_score(self):
        # Score based on per sample nuclei statistics
        nuclei = []
        for i in range(len(self.nuclei_labels)):
            for j in range(len(self.nuclei_labels[i])):
                nuclei_ = np.zeros(len(self.config.nuclei_types[1:]))
                for k in range(nuclei_.size):
                    nuclei_[k] = sum(self.nuclei_labels[i][j] == k)
                if j == 0:
                    nuclei__ = nuclei_ #/ np.sum(nuclei_)
                else:
                    nuclei__ = np.vstack((nuclei__, nuclei_ ))  # / np.sum(nuclei_)
            nuclei.append(nuclei__)

        histogram_per_tumor_type = []
        for i, nuclei_per_tumor_type in enumerate(nuclei):
            histogram = np.sum(nuclei_per_tumor_type, axis=0)
            histogram = histogram / np.sum(histogram)
            histogram_per_tumor_type.append(histogram)

        distance_per_pair = self.get_distance(histogram_per_tumor_type, self.nuclei_dist)

        return distance_per_pair
