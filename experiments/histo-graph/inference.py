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

from histocartography.utils.io import read_params
from histocartography.utils.graph import to_cpu, to_device
from histocartography.utils.io import DATATYPE_TO_EXT, DATATYPE_TO_SAVEFN
from histocartography.dataloader.pascale_dataloader import make_data_loader
from histocartography.ml.models.constants import AVAILABLE_MODEL_TYPES, MODEL_TYPE, MODEL_MODULE
from histocartography.dataloader.constants import CLASS_SPLIT_TO_MODEL_URL, TREE_CLASS_SPLIT, get_label_to_tumor_type
from histocartography.evaluation.evaluator import AccuracyEvaluator, WeightedF1
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

FOLD_IDS = [0]


def get_predictions(dataloader, mode='one-shot', model_name='cell_graph_model'):  # mode = {7_class_one_shot, 4_class_one_shot, tree}

    if mode == '7_class_one_shot':
        model = load_mlflow_model(model_name, 'benignVSpathologicalbenignVSudhVSadhVSfeaVSdcisVSmalignant')
        all_test_logits = []
        all_test_labels = []
        for data, labels in tqdm(dataloader, desc='Testing: {}'.format(epoch), unit='batch'):
            with torch.no_grad():
                labels = labels.to(DEVICE)
                logits = model(data)
            all_test_logits.append(logits)
            all_test_labels.append(labels)

        all_test_logits = torch.cat(all_test_logits).cpu()
        all_test_labels = torch.cat(all_test_labels).cpu()

    elif mode == 'tree':
        # A. load all the binary classification models 
        all_tree_models = load_tree_models(model_name)

        # B. forward pass over all the binary classification models 
        all_test_predictions = []
        all_test_labels = []
        for data, label in tqdm(dataloader, desc='Testing: {}'.format(epoch), unit='batch'):
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

            all_test_predictions.append(prediction)
            all_test_labels.append(label)

        all_test_logits = torch.cat(all_test_logits).cpu()
        all_test_labels = torch.cat(all_test_labels).cpu()

    else:
        raise NotImplementedError('Unsupported mode:', mode)
    return all_test_logits, all_test_labels


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
    model_type = config[MODEL_TYPE]
    if model_type in list(AVAILABLE_MODEL_TYPES.keys()):
        # module = importlib.import_module(
        #     MODEL_MODULE.format(model_type)
        # )
        # model = getattr(module, AVAILABLE_MODEL_TYPES[model_type])(
        #     config['model_params'], num_cell_features).to(DEVICE)
        fname = BASE_S3 + CLASS_SPLIT_TO_MODEL_URL[model_name][class_split] 
        model = load_model(fname,  map_location=torch.device('cpu'))

        # def is_int(s):
        #     try:
        #         int(s)
        #         return True
        #     except:
        #         return False

        # for n, p in mlflow_model.named_parameters():
        #     split = n.split('.')
        #     to_eval = 'model'
        #     for s in split:
        #         if is_int(s):
        #             to_eval += '[' + s + ']'
        #         else:
        #             to_eval += '.'
        #             to_eval += s
        #     exec(to_eval + '=' + 'p')

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

    # load config file
    config = read_params(args.config_fpath, verbose=True)

    if args.num_classes is not None:
        config['model_params']['num_classes'] = args.num_classes

    # mlflow log parameters
    mlflow.log_params({
        'number_of_workers': args.number_of_workers,
        'batch_size': args.batch_size
    })

    df = pd.io.json.json_normalize(config)
    rep = {"graph_building.": "", "model_params.": "", "gnn_params.": ""} # replacement for shorter key names
    for i, j in rep.items():
        df.columns = df.columns.str.replace(i, j)
    flatten_config = df.to_dict(orient='records')[0]
    for key, val in flatten_config.items():
        mlflow.log_params({key: str(val)})

    # set model path
    model_path = complete_path(args.model_path, str(uuid.uuid4()))
    check_for_dir(model_path)

    for fold_id in FOLD_IDS:

        print('Start fold: {}'.format(fold_id))

        # make data loaders (train & validation)
        dataloaders, input_feature_dims = make_data_loader(
            batch_size=args.batch_size,
            num_workers=args.number_of_workers,
            path=args.data_path,
            class_split="benignVSpathologicalbenignVSudhVSadhVSfeaVSdcisVSmalignant",  # 7-class problem default split
            config=config,
            cuda=CUDA,
            load_cell_graph=load_cell_graph(config['model_type']),
            load_superpx_graph=load_superpx_graph(config['model_type']),
            load_image=False,
            load_in_ram=args.in_ram,
            show_superpx=False,
            fold_id=fold_id
        )

        # get predictions 
        logits, labels = get_predictions(model, dataloaders['test'])

        # metrics 1: accuracy 
        accuracy = metrics['accuracy'](all_test_logits, all_test_labels).item()
        mlflow.log_metric('test_accuracy_' + str(fold_id), accuracy, step=step)

        # metrics 2: weighted F1-score 
        weighted_f1_score = metrics['weighted_f1_score'](all_test_logits, all_test_labels).item()
        mlflow.log_metric('test_weighted_f1_score_' + str(fold_id), weighted_f1_score, step=step)


if __name__ == "__main__":
    main(args=parse_arguments())
