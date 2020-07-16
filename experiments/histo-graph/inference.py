#!/usr/bin/env python3
"""
Script for training graph-based histocartography models
"""
import importlib
import torch
import mlflow
import os
import pickle
import uuid
from tqdm import tqdm
import mlflow.pytorch
import pandas as pd
import shutil
from mlflow.pytorch import load_model
import csv 

from histocartography.utils.io import read_params
from histocartography.utils.graph import to_cpu, to_device
from histocartography.utils.io import DATATYPE_TO_EXT, DATATYPE_TO_SAVEFN
from histocartography.dataloader.pascale_dataloader import make_data_loader
from histocartography.ml.models.constants import AVAILABLE_MODEL_TYPES, MODEL_TYPE, MODEL_MODULE
from histocartography.dataloader.constants import CLASS_SPLIT_TO_MODEL_URL, TREE_CLASS_SPLIT, get_label_to_tumor_type, get_tumor_type_to_label, ALL_CLASS_SPLITS
from histocartography.evaluation.evaluator import AccuracyEvaluator, WeightedF1, ExpectedClassShiftWithHardPred, ExpectedClassShiftWithLogits
from histocartography.evaluation.confusion_matrix import ConfusionMatrix
from histocartography.evaluation.classification_report import ClassificationReport, PerClassWeightedF1Score
from histocartography.utils.arg_parser import parse_arguments
from histocartography.ml.models.constants import load_superpx_graph, load_cell_graph
from histocartography.utils.io import (
    get_device, check_for_dir,
    complete_path, load_checkpoint,
    save_checkpoint
)

import warnings
warnings.filterwarnings("ignore")

# Things to change/adapt from BACH to BRACS-L:
# 1- set the default class split:
#     - BRACS-L: benignVSpathologicalbenignVSudhVSadhVSfeaVSdcisVSmalignant
#     -BACH: benignVSpathologicalbenignVSdcisVSmalignant
# 2- set the node feature type:
#     - set in the base config for BOTH TG and CG configurations
# 3- node feature type must match the ones used by the mlflow model 

# cuda support
CUDA = torch.cuda.is_available()
DEVICE = get_device(CUDA)
BASE_S3 = "s3://mlflow/"
FOLD_IDS = [0]
# DEFAULT_CLASS_SPLIT = 'benignVSpathologicalbenignVSudhVSadhVSfeaVSdcisVSmalignant'
DEFAULT_CLASS_SPLIT = 'benignVSpathologicalbenignVSdcisVSmalignant'

# if BACH challenge default split --> test only on one split 
if DEFAULT_CLASS_SPLIT == 'benignVSpathologicalbenignVSdcisVSmalignant':
    SAVE_PREDICTIONS_AS_CSV = True
    ALL_CLASS_SPLITS = [DEFAULT_CLASS_SPLIT]
    CLASS_SPLIT_TO_MODEL_URL = {
        'multi_level_graph_model': {
            'benignVSpathologicalbenignVSdcisVSmalignant': '90af5ef5976640f68a06687d71ca9319/artifacts/model_best_val_weighted_f1_score_0'
        }
    }


def get_predictions(
        dataloader,
        class_split=DEFAULT_CLASS_SPLIT,
        mode='one_shot',
        model_type='cell_graph_model'
    ):  

    if mode == 'one_shot':
        model = load_mlflow_model(model_type, class_split)
        all_test_logits = []
        all_test_labels = []
        for data, label in tqdm(dataloader, desc='Testing:', unit='batch'):
            if get_label_to_tumor_type(DEFAULT_CLASS_SPLIT)[label.item()] in class_split.replace('VS', '+').split('+'):
                with torch.no_grad():
                    label = label.to(DEVICE)
                    logits = model(data)
                all_test_logits.append(logits)
                all_test_labels.append(
                    get_tumor_type_to_label(class_split)[
                        get_label_to_tumor_type(DEFAULT_CLASS_SPLIT)[
                            label.item()
                        ]
                    ]
                )

        all_test_logits = torch.cat(all_test_logits).cpu()
        all_test_labels = torch.FloatTensor(all_test_labels)

        return all_test_logits, all_test_labels

    elif mode == 'tree':
        # A. load all the binary classification models 
        all_tree_models = load_tree_models(model_type)

        # B. forward pass over all the binary classification models 
        all_test_predictions = []
        all_test_labels = []
        for data, label in tqdm(dataloader, desc='Testing:', unit='batch'):
            with torch.no_grad():
                label = label.to(DEVICE)
                current_class_split = 'benign+pathologicalbenign+udh+adh+fea+dcisVSmalignant'
                while current_class_split is not None:

                    # 1/ get prediction for the current class split 
                    logits = all_tree_models[current_class_split](data)
                    _, prediction = torch.max(logits, dim=1)
                    prediction = get_label_to_tumor_type(current_class_split)[prediction.item()]

                    # 2/ update the class split -- stop if None 
                    current_class_split = find_next_class_split(current_class_split, logits)

            all_test_predictions.append(get_tumor_type_to_label(DEFAULT_CLASS_SPLIT)[prediction])
            all_test_labels.append(label.item())

        all_test_predictions = torch.FloatTensor(all_test_predictions).cpu()
        all_test_labels = torch.FloatTensor(all_test_labels).cpu()
        return all_test_predictions, all_test_labels

    else:
        raise NotImplementedError('Unsupported mode:', mode)


def find_next_class_split(current_class_split, logits):
    _, winning_class = torch.max(logits, dim=1)
    winning_class_name = get_label_to_tumor_type(current_class_split)[winning_class.item()]
    new_classes = [i for i in current_class_split.split('VS') if winning_class_name in i][0].split('+')
    new_class_split = [candidate_class_split for candidate_class_split in TREE_CLASS_SPLIT if
            candidate_class_split.replace('VS', '+').split('+') == new_classes]
    if new_class_split:
        return new_class_split[0]
    return None 


def load_tree_models(model_type):
    all_tree_models = {}
    for class_split in TREE_CLASS_SPLIT:
        all_tree_models[class_split] = load_mlflow_model(model_type, class_split)
    return all_tree_models


def load_mlflow_model(model_type, class_split):
    if model_type in list(AVAILABLE_MODEL_TYPES.keys()):
        fname = os.path.join(BASE_S3, CLASS_SPLIT_TO_MODEL_URL[model_type][class_split])
        print('Load model {} at location {}'.format(class_split, fname))
        model = load_model(fname,  map_location=torch.device('cpu'))
        if CUDA:
            model = model.cuda()
        return model
    else:
        raise ValueError(
            'Model: {} not recognized. Options are: {}'.format(
                model_type, list(AVAILABLE_MODEL_TYPES.keys())
            )
        )


def main(args):
    """
    Train HistoGraph.
    Args:
        args (Namespace): parsed arguments.
    """

    # Set MODEL TYPE here  
    model_type = 'multi_level_graph_model'   # 'cell_graph_model', 'multi_level_graph_model', 'concat_graph_model'

    base_config = {
        "graph_building": {
            "cell_graph_builder": {
                "edge_encoding": False,
                "graph_building_type": "knn_graph_builder",
                "max_distance": 50,
                "n_neighbors": 5,
                "node_feature_types": ["features_cnn_resnet50_mask_False_"]
            },
            "superpx_graph_builder": {
                "edge_encoding": False,
                "graph_building_type": "rag_graph_builder",
                "node_feature_types": [
                    "merging_hc_features_cnn_resnet50_mask_False_"
                ]
            }
        },
        "model_type": "cell_graph_model"
    }
    
    for fold_id in FOLD_IDS:

        print('Start fold: {}'.format(fold_id))

        # make data loaders (train & validation)
        dataloaders, input_feature_dims = make_data_loader(
            batch_size=args.batch_size,
            num_workers=args.number_of_workers,
            path=args.data_path,
            class_split=DEFAULT_CLASS_SPLIT,  # 7-class problem default split
            config=base_config,
            cuda=CUDA,
            load_cell_graph=load_cell_graph(model_type),
            load_superpx_graph=load_superpx_graph(model_type),
            load_image=False,
            load_in_ram=args.in_ram,
            show_superpx=False,
            fold_id=fold_id
        )

        # define evaluators 
        eval_expected_class_shift_with_logits = ExpectedClassShiftWithLogits()
        eval_expected_class_shift_with_hard_pred = ExpectedClassShiftWithHardPred()
        eval_accuracy = AccuracyEvaluator()
        eval_weighted_f1_score = WeightedF1()
        eval_per_class_weighted_f1_score = PerClassWeightedF1Score()

        for class_split in ALL_CLASS_SPLITS:

            # 1. get predictions (logits + labels)
            logits, labels = get_predictions(dataloaders['test'], class_split=class_split, mode='one_shot', model_type=model_type)

            # 2. get all the metric values 
            #   - metric 1: expected class shift 
            expected_class_shift = eval_expected_class_shift_with_logits(logits, labels).item()

            #   - metric 2: accuracy 
            accuracy = eval_accuracy(logits, labels).item()

            #    - metric 3: weighted F1-score 
            weighted_f1_score = eval_weighted_f1_score(logits, labels).item()

            #   - metric 4: per class weighted F1-score
            per_class_weighted_f1_score = eval_per_class_weighted_f1_score(logits, labels, class_split)

            # 3. print results 
            print('*** Model type ***', model_type)
            print('Class split', class_split)
            print('    - Expected class shift:', expected_class_shift)
            print('    - Accuracy:', accuracy)
            print('    - Weighted F1-score:', weighted_f1_score)
            print('    - Per class weighted F1-score:', per_class_weighted_f1_score)
            print('')

        # Save predictions for BACH challenge 
        if SAVE_PREDICTIONS_AS_CSV:
            _, predictions = torch.max(logits, dim=1)
            predictions = list(predictions.cpu().numpy())
            with open(CLASS_SPLIT_TO_MODEL_URL[model_type][DEFAULT_CLASS_SPLIT].split('/')[0] + '.csv', 'w', newline='') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow(predictions)
            print('Predictions are:', predictions)
            exit()

        # get tree predictions 
        predictions, labels = get_predictions(dataloaders['test'], mode='tree', model_type=model_type)

        # metrics 1: expected class shift 
        expected_class_shift = eval_expected_class_shift(predictions, labels).item()
        print('Tree classification')
        print('   - Expected class shift:', expected_class_shift)


if __name__ == "__main__":
    main(args=parse_arguments())
