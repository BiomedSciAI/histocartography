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

from histocartography.utils.io import read_params
from histocartography.utils.graph import to_cpu, to_device
from histocartography.utils.io import DATATYPE_TO_EXT, DATATYPE_TO_SAVEFN
from histocartography.dataloader.pascale_dataloader import make_data_loader
from histocartography.ml.models.constants import AVAILABLE_MODEL_TYPES, MODEL_TYPE, MODEL_MODULE
from histocartography.dataloader.constants import CLASS_SPLIT_TO_MODEL_URL, TREE_CLASS_SPLIT, get_label_to_tumor_type, get_tumor_type_to_label
from histocartography.evaluation.evaluator import AccuracyEvaluator, WeightedF1, ExpectedClassShiftWithHardPred, ExpectedClassShiftWithLogits
from histocartography.evaluation.confusion_matrix import ConfusionMatrix
from histocartography.evaluation.classification_report import ClassificationReport
from histocartography.utils.arg_parser import parse_arguments
from histocartography.ml.models.constants import load_superpx_graph, load_cell_graph
from histocartography.utils.io import (
    get_device, check_for_dir,
    complete_path, load_checkpoint,
    save_checkpoint
)

import warnings
warnings.filterwarnings("ignore")

# cuda support
CUDA = torch.cuda.is_available()
DEVICE = get_device(CUDA)
BASE_S3 = "s3://mlflow/"
FOLD_IDS = [0]


def get_predictions(dataloader, mode='one_shot', model_name='cell_graph_model'):  # mode = {7_class_one_shot, 4_class_one_shot, tree}

    if mode == 'one_shot':
        model = load_mlflow_model(model_name, 'benignVSpathologicalbenignVSudhVSadhVSfeaVSdcisVSmalignant')
        all_test_logits = []
        all_test_labels = []
        for data, labels in tqdm(dataloader, desc='Testing:', unit='batch'):
            with torch.no_grad():
                labels = labels.to(DEVICE)
                logits = model(data)
            all_test_logits.append(logits)
            all_test_labels.append(labels)

        all_test_logits = torch.cat(all_test_logits).cpu().to(float)
        all_test_labels = torch.cat(all_test_labels).cpu().to(float)

        return all_test_logits, all_test_labels

    elif mode == 'tree':
        # A. load all the binary classification models 
        all_tree_models = load_tree_models(model_name)

        # B. forward pass over all the binary classification models 
        all_test_predictions = []
        all_test_labels = []
        counter = 0
        for data, label in tqdm(dataloader, desc='Testing:', unit='batch'):
            counter += 1
            with torch.no_grad():
                label = label.to(DEVICE)
                current_class_split = 'benign+pathologicalbenign+udh+adh+fea+dcisVSmalignant'
                while current_class_split is not None:
                    # 1/ get prediction for the current class split 
                    # print('Forward pass with class split:', current_class_split)
                    # print('Label:', get_label_to_tumor_type('benignVSpathologicalbenignVSudhVSadhVSfeaVSdcisVSmalignant')[label.item()])
                    logits = all_tree_models[current_class_split](data)
                    _, prediction = torch.max(logits, dim=1)
                    prediction = get_label_to_tumor_type(current_class_split)[prediction.item()]
                    # print('Prediction:', prediction)
                    # 2/ update the class split -- stop if None 
                    current_class_split = find_next_class_split(current_class_split, logits)
            # # debug purposes 
            # if counter >= 20:
            #     break 
            all_test_predictions.append(get_tumor_type_to_label('benignVSpathologicalbenignVSudhVSadhVSfeaVSdcisVSmalignant')[prediction])
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


def load_tree_models(model_name):
    all_tree_models = {}
    for class_split in TREE_CLASS_SPLIT:
        all_tree_models[class_split] = load_mlflow_model(model_name, class_split)
    return all_tree_models


def load_mlflow_model(model_name, class_split):
    if model_name in list(AVAILABLE_MODEL_TYPES.keys()):
        fname = os.path.join(BASE_S3, CLASS_SPLIT_TO_MODEL_URL[model_name][class_split])
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

    # @TODO: hardcode required input parameters 
    model_type = 'cell_graph_model' 

    base_config = {
        "graph_building": {
            "cell_graph_builder": {
                "edge_encoding": False,
                "graph_building_type": "knn_graph_builder",
                "max_distance": 50,
                "n_neighbors": 5,
                "node_feature_types": ["features_cnn_resnet34_mask_False_"]
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
            class_split="benignVSpathologicalbenignVSudhVSadhVSfeaVSdcisVSmalignant",  # 7-class problem default split
            config=base_config,
            cuda=CUDA,
            load_cell_graph=load_cell_graph(model_type),
            load_superpx_graph=load_superpx_graph(model_type),
            load_image=False,
            load_in_ram=args.in_ram,
            show_superpx=False,
            fold_id=fold_id
        )

        # get 7-class predictions 
        logits, labels = get_predictions(dataloaders['test'], mode='one_shot', model_name=model_type)
        # metrics 1: expected class shift 
        eval_expected_class_shift = ExpectedClassShiftWithLogits()
        expected_class_shift = eval_expected_class_shift(logits, labels).item()
        print('7-class // Expected class shift:', expected_class_shift)

        # get tree predictions 
        predictions, labels = get_predictions(dataloaders['test'], mode='tree', model_name=model_type)
        # metrics 1: expected class shift 
        eval_expected_class_shift = ExpectedClassShiftWithHardPred()
        expected_class_shift = eval_expected_class_shift(predictions, labels).item()
        print('Tree // Expected class shift:', expected_class_shift)

        # metrics 1: accuracy 
        # accuracy = metrics['accuracy'](all_test_logits, all_test_labels).item()
        # print('Accuracy:', accuracy)
        # mlflow.log_metric('test_accuracy_' + str(fold_id), accuracy, step=step)

        # metrics 2: weighted F1-score 
        # weighted_f1_score = metrics['weighted_f1_score'](all_test_logits, all_test_labels).item()
        # print('Weighted F1-score:', weighted_f1_score)
        # mlflow.log_metric('test_weighted_f1_score_' + str(fold_id), weighted_f1_score, step=step)


if __name__ == "__main__":
    main(args=parse_arguments())
