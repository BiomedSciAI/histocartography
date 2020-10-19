import copy
import numpy as np
from matplotlib import pyplot as plt
from utils import *


def plot_concept_map_per_tumor_type(args, config, explainer, percentage, explanation, xlim=[-0.05, 1.05], ylim=[-0.05, 1.05]):
    fig, axes = plt.subplots(len(explanation.node_importance))
    fig.suptitle(explainer + ' : ' + args.concept + ' : ' + str(round(percentage, 2)))

    for t in range(len(explanation.node_importance)):
        importance = np.array([])
        concept = np.array([])

        for i in range(len(explanation.node_importance[t])):
            importance = np.append(importance, explanation.node_importance[t][i])
            concept = np.append(concept, explanation.node_concept[t][i])

        color = [config.tumor_colors[t]] * len(importance)

        axes[t].set_title(config.tumor_types[t])
        axes[t].scatter(concept, importance, c=color, alpha=0.3, edgecolors='none')
        axes[t].grid(True)
        axes[t].set_xlim(xlim)
        axes[t].set_ylim(ylim)

        if t != len(explanation.node_importance) - 1:
            axes[t].get_xaxis().set_visible(False)

    plt.savefig(config.figure_save_path + explainer + ' - ' + args.concept + '_' + str(percentage, 2) + '_per_tumor_type.png', dpi=300)
    plt.close()


def plot_concept_map_per_tumor_class(args, config, explainer, percentage, explanation, xlim=[-0.05, 1.05], ylim=[-0.05, 1.05]):
    n_tumors = len(np.unique(config.tumor_labels))

    fig, axes = plt.subplots(n_tumors)
    fig.suptitle(explainer + ' : ' + args.concept + ' : ' + str(round(percentage, 2)))

    for t in range(n_tumors):
        idx = np.where(config.tumor_labels == t)[0]

        importance = np.array([])
        concept = np.array([])

        color = ''
        label = -1
        for id in idx:
            color = config.tumor_colors[id]
            label = config.tumor_labels[id]
            for i in range(len(explanation.node_importance[id])):
                importance = np.append(importance, explanation.node_importance[id][i])
                concept = np.append(concept, explanation.node_concept[id][i])
        color = [color] * len(importance)

        axes[t].set_title('Class: ' + str(label))
        axes[t].scatter(concept, importance, c=color, alpha=0.3, edgecolors='none')
        axes[t].grid(True)
        axes[t].set_xlim(xlim)
        axes[t].set_ylim(ylim)

    plt.savefig(config.figure_save_path + explainer + ' - ' + args.concept + '_' + str(round(percentage, 2)) + '_per_tumor_class.png', dpi=300)
    plt.close()


def plot_auc_map(args, config, p_scores):
    fill_colors = ['chartreuse', 'gold', 'violet', 'coral']
    line_colors = ['green', 'darkgoldenrod', 'purple', 'red']
    plt.title('Score vs Percentage: ' + args.concept)
    for i in range(len(config.explainers)):
        plt.plot(config.percentages, p_scores[i], 'ko')
        plt.plot(config.percentages, p_scores[i], line_colors[i], label=config.explainers[i])
        plt.fill_between(config.percentages, p_scores[i], color=fill_colors[i])  # alpha=0.3

    plt.legend()
    plt.xlabel('Percentage')
    plt.ylabel('Score')

    name = ''
    for e in config.explainers:
        name += e + '_'

    plt.savefig(config.figure_save_path + name + args.concept + '.png', dpi=300)
    plt.close()


def plot_scatter(node_importance, node_concept):
    fig, ax = plt.subplots(1, 2)

    ctr = 0
    for importance, concept in zip(node_importance, node_concept):
        color = []
        max_val = max(importance)
        min_val = min(importance)
        for i in range(len(importance)):
            fill = rgb(min_val, max_val, importance[i])
            color.append(fill)
        ax[ctr].scatter(concept, importance, c=color)
        ctr += 1
    plt.show()


