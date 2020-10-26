import argparse
from config import *
from explainability import *
from metric import *
from sklearn.metrics import auc
from plotting import *

parser = argparse.ArgumentParser()
parser.add_argument('--explainer',
                    help='Explainability method',
                    choices=['GraphLRP', 'GraphGradCAM', 'GNNExplainer', 'GraphGradCAMpp', '-1'],
                    required=True)
parser.add_argument('--base-path',
                    help='Base path to the data folder',
                    required=False)
parser.add_argument('--nuclei-selection-type',
                    help='Nuclei selection type, eg. w/ hard thresholding, w/ cumulutative',
                    choices=['cumul', 'thresh', 'absolute'],
                    default='absolute',
                    required=False)
parser.add_argument('--classification-mode',
                    help='Classification mode',
                    choices=[2, 3, 5, 7],
                    default=3,
                    type=int,
                    required=False)
parser.add_argument('--concept',
                    help='Node concept to analyze', required=True)
parser.add_argument('--p',
                    help='Node importance > p to keep',
                    type=float,
                    default=-1,
                    required=False)
parser.add_argument('--distance',
                    help='Point cloud distance measure',
                    choices=['pair', 'chamfer', 'hausdorff'],
                    default='chamfer',
                    required=False)
parser.add_argument('--risk',
                    help='With class-shift risk',
                    default='True',
                    required=False)
parser.add_argument('--verbose',
                    help='Verbose flag',
                    default='False',
                    required=False)
parser.add_argument('--visualize',
                    help='Visualize flag',
                    default='False',
                    required=False)
parser.add_argument('--rm_misclassification',
                    help='If we should filter out misclassified samples.',
                    default='True',
                    required=False)
parser.add_argument('--rm-non-epithelial-nuclei',
                    help='If we should remove all the non epithial nuclei.',
                    default='False',
                    required=False)
parser.add_argument('--with-nuclei-selection-plot',
                    help='If we should plot the nuclei selection along with the image for each sample.',
                    default='False',
                    required=False)
parser.add_argument('--extract_features',
                    help='If we should extract nuclei features',
                    default='False',
                    required=False)

args = parser.parse_args()
config = Configuration(args=args)
args.concept = args.concept.split(',')


# *************************************************************************** Set parameters
verbose = eval(args.verbose)
visualize = eval(args.visualize)
rm_misclassification = eval(args.rm_misclassification)
with_nuclei_selection_plot = eval(args.with_nuclei_selection_plot)
args.rm_non_epithelial_nuclei = eval(args.rm_non_epithelial_nuclei)

percentages = config.percentages
explainers = config.explainers

# Get TRoI sample names
config.get_sample_names(args, explainers)
print('Total #TRoI: ', len(config.samples))


# *************************************************************************** Extract features
if eval(args.extract_features):
    from extract_features import *
    extract = ExtractFeatures(config)
    extract.extract_feature()


# *************************************************************************** Get explanation
p_concept_scores = []
p_nuclei_scores = []

for e in explainers:
    print('\n********************************************')
    print('Explainer: ', e)
    concept_scores = np.array([])
    nuclei_scores = np.array([])
    precision_epi_scores = np.array([])

    for p in percentages:
        exp = Explainability(
            args=args,
            config=config,
            explainer=e,
            percentage=p,
            verbose=verbose,
            visualize=visualize
        )
        exp.get_explanation(rm_misclassification)

        # plot nuclei selection on the original image 
        if with_nuclei_selection_plot:
            plot_nuclei_selection(exp, base_path=args.base_path)

        m = Metric(args=args, config=config, explainer=e, percentage=p, explanation=exp)
        concept_score = m.compute_concept_score()
        concept_scores = np.append(concept_scores, concept_score)

        precision_epi, nuclei_score = m.compute_nuclei_score()
        # precision_epi_scores = np.append(precision_epi_scores, precision_epi)
        nuclei_scores = np.append(nuclei_scores, nuclei_score)

        print(
            'p= ',
            round(p, 2),
            ' --nTRoI: ',
            np.sum(exp.samples),
            ' --nNodes: ',
            len(exp.labels),
            ' --concept-score= ',
            concept_score,
            ' --nuclei-score= ',
            nuclei_score,
            # ' --precision-epi= ',
            # precision_epi_scores
            )

        if visualize:
            #plot_concept_map_per_tumor_type(args, config, e, p, exp)
            plot_concept_map_per_tumor_class(args, config, e, p, exp)

    p_concept_scores.append(concept_scores)
    p_nuclei_scores.append(nuclei_scores)


if visualize:
    plot_auc_map(args, config, p_concept_scores, title='Concept score vs Percentage: ' + args.concept, filename='concept')
    plot_auc_map(args, config, p_nuclei_scores, title='Nuclei score vs Percentage: ' + args.concept, filename='nuclei')





