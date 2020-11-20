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

        # Create D-dimensional histogram
        count = []
        for i in range(len(input)):
            H, _ = np.histogramdd(input[i], bins=bins, density=True)
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
        
        # 1. get all pairs of classes 
        all_pairs = list(itertools.combinations(self.classes, 2))
        distance_per_pair = {}

        # 2. compute distance for each pair 
        for pair in all_pairs:
            score = dist.distance(input[pair[0]], input[pair[1]])
            distance_per_pair[str(pair)] = score

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
                    '''
                    mask = self.nuclei_labels[i][j] == k
                    if isinstance(self.nuclei_labels[i][j], np.float64):
                        nuclei_[k] = mask
                    else:
                        nuclei_[k] = sum(mask)
                    #'''

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
            # print('Histogram:', i, histogram)

        # # Histogram analysis
        # all_nuclei_histograms = []
        # for nuclei_type in range(nuclei[0].shape[1]):  # 0 to 5
        #     nuclei_type_data = [nuclei[tumor_type][:, nuclei_type, None] for tumor_type in range(len(nuclei))]
        #     histogram = self.histogram_analysis(nuclei_type_data, step=0.05)
        #     all_nuclei_histograms.append(histogram)
        
        # all_distances_per_pair = [self.get_distance(x, self.nuclei_dist) for x in all_nuclei_histograms]
        # merged_distance_per_pair = {}
        # for pair in all_distances_per_pair[0].keys():
        #     merged_distance_per_pair[pair] = sum([val[pair] for val in all_distances_per_pair])
    
        distance_per_pair = self.get_distance(histogram_per_tumor_type, self.nuclei_dist) 
            
        return distance_per_pair


# concepts are organised as: 'type', 'roundness', 'ellipticity', 'crowdedness', 'std_h', 'area'
# class pairs are organised as: (0, 1), (0, 2), (1, 2)

PATHOLOGIST_CONCEPT_RANKING = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3],
    [4, 4, 4],
    [5, 5, 5]
])


class ConceptRanking:
    def __init__(self, score_per_concept_per_percentage_per_pair):

        self.score_per_concept_per_percentage_per_pair = score_per_concept_per_percentage_per_pair

        # extract all concepts
        self.all_concepts = list(self.score_per_concept_per_percentage_per_pair.keys())
        self.num_concepts = len(self.all_concepts)

        # extract all percentages (p)
        self.all_p = list(self.score_per_concept_per_percentage_per_pair[self.all_concepts[0]].keys())
        self.num_p = len(self.all_p)

        # extract all class pairs 
        self.all_pairs = list(self.score_per_concept_per_percentage_per_pair[self.all_concepts[0]][self.all_p[0]].keys())
        self.num_pairs = len(self.all_pairs)

        self._compute_scores_as_matrix()

    def rank_concepts(self, p_to_keep='auc', with_risk=True):
        scores_to_keep = self.concept_scores[:, self.all_p.index(p_to_keep), :]  # dim: num_concepts x num_pairs
        scores_ranked = np.argsort(-scores_to_keep, axis=0)  # rank from larger concept value to smallest 

        ranking_score_per_pair = {}
        for pair_id, pair in enumerate(self.all_pairs):
            ranking_score_per_pair[pair] = stats.spearmanr(scores_ranked[:, pair_id], PATHOLOGIST_CONCEPT_RANKING[:, pair_id])[0]
            # print('Pair:', pair, ' | Ranking are:', scores_ranked[:, pair_id], ' and ', PATHOLOGIST_CONCEPT_RANKING[:, pair_id])
            # print('With correlation:', ranking_score_per_pair[pair])

        if with_risk:
            risk = self.get_risk_per_pair()
        else:
            risk = [1] * self.num_pairs

        aggregated_ranking_score = sum([val * risk[id] for id, (_, val) in enumerate(ranking_score_per_pair.items())]) / self.num_pairs

        return ranking_score_per_pair, aggregated_ranking_score

    def get_risk_per_pair(self):
        risk_per_pair = [np.abs(pair[1] - pair[0]) for pair in self.all_pairs]
        return risk_per_pair

    def _compute_scores_as_matrix(self):
        self.concept_scores = np.zeros((self.num_concepts, self.num_p, self.num_pairs))
        for concept_id, (_, concept_val) in enumerate(self.score_per_concept_per_percentage_per_pair.items()):
            for p_id, (_, p_val) in enumerate(concept_val.items()):
                for pair_id, (_, pair_val) in enumerate(p_val.items()):
                    self.concept_scores[concept_id, p_id, pair_id] = pair_val
