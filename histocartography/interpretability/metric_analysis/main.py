def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import numpy as np
from config import *
from explainability import *
from concept_metrics import ConceptMetric
from sklearn.metrics import auc
from plotting import *
from histocartography.utils.io import write_json


# ALL_CONCEPTS = ['roundness', 'ellipticity', 'crowdedness', 'std_h', 'area', ]
ALL_CONCEPTS = [
    'area',               #
    'perimeter',          #
    'roughness',          #
    'eccentricity',       #
    'roundness',          #
    'shape_factor',       #
    'crowdedness',        #
    'std_crowdedness',    #
    'glcm_dissimilarity', #   
    'std_h',              # @TODO: is it the same as contrast?
    'glcm_homogeneity',   #
    'glcm_ASM',           #
    'glcm_entropy',       #
    'glcm_variance'       #
]

parser = argparse.ArgumentParser()
parser.add_argument('--explainer',
                    help='Explainability method',
                    choices=['GraphLRP', 'GraphGradCAM', 'GNNExplainer', 'GraphGradCAMpp', '-1'],
                    required=True)
parser.add_argument('--base-path',
                    help='Base path to the data folder',
                    required=False)
parser.add_argument('--classification-mode',
                    help='Classification mode',
                    choices=[2, 3],
                    default=3,
                    type=int,
                    required=False)
parser.add_argument('--extract_features',
                    help='If we should extract nuclei features',
                    default='False',
                    required=False)
# parser.add_argument('--concept',
#                     help='Node concept to analyze', required=True)
parser.add_argument('--p',
                    help='Node importance > p to keep',
                    type=float,
                    default=-1,
                    required=False)
parser.add_argument('--distance',
                    help='Point cloud distance measure',
                    choices=['pair', 'chamfer', 'hausdorff', 'svm', 'hist', 'wassertein'],
                    default='hist',
                    required=False)
parser.add_argument('--nuclei-selection-type',
                    help='Nuclei selection type, eg. w/ hard thresholding, w/ cumulutative',
                    choices=['cumul', 'thresh', 'absolute', 'random'],
                    default='absolute',
                    required=False)
parser.add_argument('--rm-non-epithelial-nuclei',
                    help='If we should remove all the non epithelial nuclei.',
                    default='False',
                    required=False)
parser.add_argument('--risk',
                    help='With class-shift risk',
                    default='True',
                    required=False)
parser.add_argument('--rm-misclassification',
                    help='If we should filter out misclassified samples.',
                    default='True',
                    required=False)
parser.add_argument('--with-nuclei-selection-plot',
                    help='If we should plot the nuclei selection along with the image for each sample.',
                    default='False',
                    required=False)
parser.add_argument('--verbose',
                    help='Verbose flag',
                    default='False',
                    required=False)
parser.add_argument('--visualize',
                    help='Visualize flag',
                    default='False',
                    required=False)
parser.add_argument('--tumor_type',
                    help='Explainability method',
                    choices=['adh', 'benign', 'dcis', 'fea', 'malignant', 'pathologicalbenign', 'udh'],
                    default='adh',
                    required=False)

args = parser.parse_args()
config = Configuration(args=args)
# args.concept = args.concept.split(',')

# *************************************************************************** Set parameters
verbose = eval(args.verbose)
visualize = eval(args.visualize)
args.rm_misclassification = eval(args.rm_misclassification)
args.rm_non_epithelial_nuclei = eval(args.rm_non_epithelial_nuclei)
args.with_nuclei_selection_plot = eval(args.with_nuclei_selection_plot)
percentages = config.percentages
explainers = config.explainers

# Get TRoI sample names
config.get_sample_names(args, explainers)
print('Total #TRoI: ', len(config.samples))

# *************************************************************************** Extract features
if eval(args.extract_features):
    from extract_features import *
    extract = ExtractFeatures(config)
    extract.extract_feature(tumor_types=[args.tumor_type])
    exit()

# *************************************************************************** Get explanation
p_concept_scores = []
p_nuclei_scores = []

for e in explainers:
    print('\n********************************************')
    print('Explainer: ', e)
    score_per_concept_per_percentage_per_pair = {}
    stats_per_concept_per_percentage_per_tumor_type = {}

    for concept in ALL_CONCEPTS:
        score_per_concept_per_percentage_per_pair[concept] = {}
        stats_per_concept_per_percentage_per_tumor_type[concept] = {}
        for p in percentages:
            # @TODO: create 3 different splits 
            exp = Explainability(
                args=args,
                config=config,
                explainer=e,
                concept_name=concept,
                percentage=p,
                verbose=verbose,
                visualize=visualize
            )
            exp.get_explanation()

            m = ConceptMetric(args=args, config=config, explainer=e, percentage=p, explanation=exp)
            concept_stats_per_tumor_type = m.compute_tumor_type_stats()
            concept_score_per_pair = m.compute_concept_score()
            score_per_concept_per_percentage_per_pair[concept][str(p)] = concept_score_per_pair
            stats_per_concept_per_percentage_per_tumor_type[concept][str(p)] = concept_stats_per_tumor_type

            print(
                'Concept= ',
                concept,
                'p= ',
                round(p, 2),
                ' --nTRoI: ',
                np.sum(exp.samples),
                ' --nNodes: ',
                len(exp.labels),
                ' --concept-score= ',
                concept_score_per_pair,
                )

        # compute AUC over the values of p for a given concept and for each pair of classes 
        all_pairs = [pair for pair, _ in concept_score_per_pair.items()]
        score_per_concept_per_percentage_per_pair[concept]['auc'] = {}
        for pair in all_pairs:  # loop over all the pairs
            auc_score1 = auc(percentages,
                            [score_per_concept_per_percentage_per_pair[concept][str(p)][pair][0] for p in percentages])

            auc_score2 = auc(percentages,
                            [score_per_concept_per_percentage_per_pair[concept][str(p)][pair][1] for p in percentages])

            auc_score3 = auc(percentages,
                            [score_per_concept_per_percentage_per_pair[concept][str(p)][pair][2] for p in percentages])

            score_per_concept_per_percentage_per_pair[concept]['auc'][pair] = [auc_score1, auc_score2, auc_score3]

        # # compute average over the values of p for a given concept and for each tumor type 
        # all_tumor_types = [t for t, _ in concept_stats_per_tumor_type.items()]
        # stats_per_concept_per_percentage_per_tumor_type[concept]['avg'] = {}
        # for t in all_tumor_types:  # loop over all the tumor types
        #     avg_mean = sum([stats_per_concept_per_percentage_per_tumor_type[concept][str(p)][t]['mean'] for p in percentages]) / len(percentages)
        #     avg_std = sum([stats_per_concept_per_percentage_per_tumor_type[concept][str(p)][t]['std'] for p in percentages]) / len(percentages)
        #     avg_ratio = sum([stats_per_concept_per_percentage_per_tumor_type[concept][str(p)][t]['ratio'] for p in percentages]) / len(percentages)
        #     stats_per_concept_per_percentage_per_tumor_type[concept]['avg'][t] = {
        #         'mean': float(np.round(avg_mean, 4)),
        #         'std': float(np.round(avg_std, 4)),
        #         'ratio': float(np.round(avg_ratio, 4))
        #     }

    # print & save the scores 
    # for concept_id, (concept_name, concept_val) in enumerate(score_per_concept_per_percentage_per_pair.items()):
    #     print('*** - Concept: {} | ({}/{})'.format(concept_name, concept_id + 1, len(ALL_CONCEPTS)))
    #     for p_id, (p_name, p_val) in enumerate(concept_val.items()):
    #         print('    *** - Percentage: {} | ({}/{})'.format(p_name, p_id + 1, len(percentages)))
    #         for _, (pair_name, pair_val) in enumerate(p_val.items()):
    #             print('        - Class pair: {} with distance: {}'.format(pair_name, pair_val))
    #     print('\n\n')

    print('***Split1')
    for concept_id, (concept_name, concept_val) in enumerate(score_per_concept_per_percentage_per_pair.items()):
        # print('*** - Concept: {} | ({}/{})'.format(concept_name, concept_id + 1, len(ALL_CONCEPTS)))
        # print('    *** - Percentage: {} | ({}/{})'.format(p_name, p_id + 1, len(percentages)))
        out = [np.round(pair_val[1], 4) for _, pair_val in enumerate(concept_val['auc'].items())]
        print(out[0][0], out[1][0], out[2][0])
        # for _, (pair_name, pair_val) in enumerate(p_val.items()):
        #     print('        - Class pair: {} with distance: {}'.format(pair_name, pair_val))

    print('***Split2')
    for concept_id, (concept_name, concept_val) in enumerate(score_per_concept_per_percentage_per_pair.items()):
        # print('*** - Concept: {} | ({}/{})'.format(concept_name, concept_id + 1, len(ALL_CONCEPTS)))
        # print('    *** - Percentage: {} | ({}/{})'.format(p_name, p_id + 1, len(percentages)))
        out = [np.round(pair_val[1], 4) for _, pair_val in enumerate(concept_val['auc'].items())]
        print(out[0][1], out[1][1], out[2][1])
        # for _, (pair_name, pair_val) in enumerate(p_val.items()):
        #     print('        - Class pair: {} with distance: {}'.format(pair_name, pair_val))

    print('***Split3')
    for concept_id, (concept_name, concept_val) in enumerate(score_per_concept_per_percentage_per_pair.items()):
        # print('*** - Concept: {} | ({}/{})'.format(concept_name, concept_id + 1, len(ALL_CONCEPTS)))
        # print('    *** - Percentage: {} | ({}/{})'.format(p_name, p_id + 1, len(percentages)))
        out = [np.round(pair_val[1], 4) for _, pair_val in enumerate(concept_val['auc'].items())]
        print(out[0][2], out[1][2], out[2][2])
        # for _, (pair_name, pair_val) in enumerate(p_val.items()):
        #     print('        - Class pair: {} with distance: {}'.format(pair_name, pair_val))

    write_json(e + '_output_pair.json', score_per_concept_per_percentage_per_pair)
    write_json(e + '_output_tumor_stats.json', stats_per_concept_per_percentage_per_tumor_type)
