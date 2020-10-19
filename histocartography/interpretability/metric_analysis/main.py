import argparse
from config import *
from explainability import *
from metric import *
from sklearn.metrics import auc
from plotting import *

parser = argparse.ArgumentParser()
parser.add_argument('--explainer',
                    help='Explainability method',
                    choices=['GraphLRP', 'GraphGradCAM', 'GNNExplainer', 'GraphGradCAMppExplainer', '-1'],
                    required=True)
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
parser.add_argument('--similarity',
                    help='Point cloud similarity measure',
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
                    default='True',
                    required=False)

args = parser.parse_args()
config = Configuration(args=args)


# *************************************************************************** Set parameters
verbose = eval(args.verbose)
visualize = eval(args.visualize)

percentages = config.percentages
explainers = config.explainers

# Get TRoI sample names
config.get_sample_names(args, explainers)
print('Total #TRoI: ', len(config.samples))

# *************************************************************************** Get explanation
p_scores = []

for e in explainers:
    print('\n********************************************')
    print('Explainer: ', e)
    scores = np.array([])

    for p in percentages:
        exp = Explainability(args=args, config=config, explainer=e, percentage=p, verbose=verbose, visualize=visualize)
        exp.get_explanation()

        m = Metric(args=args, config=config, explainer=e, percentage=p, explanation=exp)
        score = m.compute_score()
        scores = np.append(scores, score)
        print('p= ', round(p, 2), ' --nTRoI: ', np.sum(exp.samples), ' --nNodes: ', len(exp.labels), ' --score= ', score)

        if visualize:
            #plot_concept_map_per_tumor_type(args, config, e, p, exp)
            plot_concept_map_per_tumor_class(args, config, e, p, exp)

    p_scores.append(scores)

if visualize:
    plot_auc_map(args, config, p_scores)





