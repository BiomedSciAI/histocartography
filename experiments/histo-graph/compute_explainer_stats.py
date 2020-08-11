#!/usr/bin/env python3
"""
Script for computing dataset statistics, e.g., avg number of node per class, image size
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report

from histocartography.utils.io import read_params, get_files_in_folder, complete_path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_path',
        type=str,
        help='path to the data.',
        default='../../data/graphs',
        required=False
    )

    return parser.parse_args()


def cross_entropy_loss(target, probs):
    # cross_entropy = target_probs * torch.log(probs + 10e-10)
    cross_entropy = torch.log(probs[target] + 10e-10)
    return -torch.sum(cross_entropy)


def main(args):
    """
    Compute the explainer stats.
    Args:
        args (Namespace): parsed arguments.
    """

    # 1. get the list of JSON files
    json_fnames = get_files_in_folder(args.data_path, 'json')

    all_node_reductions = []
    all_edge_reductions = []
    all_original_cross_entropies = []
    all_explanation_cross_entropies = []
    all_random_cross_entropies = []
    all_labels = []
    all_predictions = []

    for fname in json_fnames:

        # read data
        data = read_params(complete_path(args.data_path, fname))

        # node reduction
        original_num_nodes = data['output']['original']['number_of_nodes']
        explain_num_nodes = data['output']['explanation']['number_of_nodes']
        all_node_reductions.append(float(explain_num_nodes) / float(original_num_nodes))

        # edge reduction
        original_num_edges = data['output']['original']['number_of_edges']
        explain_num_edges = data['output']['explanation']['number_of_edges']
        all_edge_reductions.append(float(explain_num_edges) / float(original_num_edges))

        # label
        label = data['output']['label_index']
        all_labels.append(label)

        # prediction
        original_probs = torch.FloatTensor(data['output']['original']['logits'])
        pred_label = torch.argmax(original_probs)
        all_predictions.append(pred_label.item())

        # original cross entropy
        all_original_cross_entropies.append(
            cross_entropy_loss(
                label, original_probs
            )
        )

        # explanation cross entropy
        explain_probs = torch.FloatTensor(data['output']['explanation']['logits'])
        all_explanation_cross_entropies.append(
            cross_entropy_loss(
                label, explain_probs
            )
        )

        # random selection cross entropy
        random_probs = torch.FloatTensor(data['output']['random']['res'][0]['logits'])
        all_random_cross_entropies.append(
            cross_entropy_loss(
                label, random_probs
            )
        )

    # 2. convert list to numpy array
    all_node_reductions = np.array(all_node_reductions)
    all_edge_reductions = np.array(all_edge_reductions)
    all_original_cross_entropies = np.array(all_original_cross_entropies)
    all_explanation_cross_entropies = np.array(all_explanation_cross_entropies)
    all_random_cross_entropies = np.array(all_random_cross_entropies)
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # 3. Overall accuracy/metrics
    print('CLASS: ALL')
    print('- Average node reduction:', 1 - np.mean(all_node_reductions), ' +/- ', np.std(all_node_reductions))
    print('- Average edge reduction:', 1 - np.mean(all_edge_reductions), ' +/- ', np.std(all_edge_reductions))
    print('- Average original cross entropy:', np.mean(all_original_cross_entropies), ' +/- ', np.std(all_original_cross_entropies))
    print('- Average explanation cross entropy:', np.mean(all_explanation_cross_entropies), ' +/- ', np.std(all_explanation_cross_entropies))
    print('- Average random cross entropy:', np.mean(all_random_cross_entropies), ' +/- ', np.std(all_random_cross_entropies))
    print('- Weighted F1-score:', f1_score(all_labels, all_predictions, average='weighted'), '\n')

    # 4. Per-class accuracy/metrics
    for cls_id in range(len(data['output']['label_set'])):
        per_class_labels = all_labels == cls_id
        per_class_node_reduction = all_node_reductions[per_class_labels]
        per_class_edge_reduction = all_edge_reductions[per_class_labels]
        per_class_original_cross_entropies = all_original_cross_entropies[per_class_labels]
        per_class_explanation_cross_entropies = all_explanation_cross_entropies[per_class_labels]
        per_class_random_cross_entropies = all_random_cross_entropies[per_class_labels]

        print('CLASS:', cls_id)
        print('- Average node reduction:', 1 - np.mean(per_class_node_reduction), ' +/- ', np.std(per_class_node_reduction))
        print('- Average edge reduction:', 1 - np.mean(per_class_edge_reduction), ' +/- ', np.std(per_class_edge_reduction))
        print('- Average original cross entropy:', np.mean(per_class_original_cross_entropies), ' +/- ', np.std(per_class_original_cross_entropies))
        print('- Average explanation cross entropy:', np.mean(per_class_explanation_cross_entropies), ' +/- ', np.std(per_class_explanation_cross_entropies))
        print('- Average random cross entropy:', np.mean(per_class_random_cross_entropies), ' +/- ', np.std(per_class_random_cross_entropies))

    # 4 classification report
    print("Classification report (for the per-class F1-score):", classification_report(all_labels, all_predictions))


if __name__ == "__main__":
    main(args=parse_arguments())
