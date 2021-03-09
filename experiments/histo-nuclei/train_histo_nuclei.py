#!/usr/bin/env python3
"""
Script for training histocartography nuclei 
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
from histocartography.dataloader.cell_dataloader import make_data_loader
from histocartography.evaluation.evaluator import AccuracyEvaluator, WeightedF1
from histocartography.evaluation.confusion_matrix import ConfusionMatrix
from histocartography.evaluation.classification_report import ClassificationReport, PerClassWeightedF1Score
from arg_parser import parse_arguments
from histocartography.ml.layers.multi_layer_gnn import MultiLayerGNN
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


def main(args):
    """
    Train HistoGraph.
    Args:
        args (Namespace): parsed arguments.
    """

    # define config parameters 
    config = {
        "model_params": {
            "activation": "relu",
            "cat": True,
            "dropout": 0.0,
            "gnn_params": {
                "cell_gnn": {
                    "activation": "relu",
                    "hidden_dim": 64,
                    "layer_type": "gin_layer",
                    "n_layers": 3,
                    "neighbor_pooling_type": "mean",
                    "output_dim": 6,
                    "return_last_layer": False
                }
            },
            "num_classes": 6,
            "use_bn": True
        },
    }

    # mlflow log parameters
    mlflow.log_params({
        'number_of_workers': args.number_of_workers,
        'batch_size': args.batch_size
    })

    # set model path
    model_path = complete_path(args.model_path, str(uuid.uuid4()))
    check_for_dir(model_path)

    # make data loaders (train & validation)
    dataloaders, input_feature_dims = make_data_loader(
        batch_size=args.batch_size,
        num_workers=args.number_of_workers,
        path=args.data_path,
        config=config,
        cuda=CUDA,
        load_in_ram=args.in_ram,
    )

    # declare model
    model = MultiLayerGNN(config['model_params']['gnn_params']['cell_gnn'])

    # build optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=5e-4
    )

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # define metrics
    accuracy_evaluation = AccuracyEvaluator(cuda=CUDA)
    weighted_f1_score = WeightedF1(cuda=CUDA)
    conf_matrix = ConfusionMatrix(return_img=True)
    class_report = ClassificationReport()
    per_class_weighted_f1_score = PerClassWeightedF1Score()
    metrics = {
        'accuracy': accuracy_evaluation,
        'weighted_f1_score': weighted_f1_score,
    }
    evaluators = {
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

    # training loop
    step = 0
    best_val_loss = 10e5
    best_val_accuracy = 0.
    best_val_weighted_f1_score = 0.

    for epoch in range(args.epochs):
        # A.) train for 1 epoch
        torch.cuda.empty_cache()
        model = model.to(DEVICE)
        model.train()
        for cell_graphs in tqdm(dataloaders['train'], desc='Epoch training {}'.format(epoch), unit='batch'):

            # 1. forward pass
            logits = model(cell_graphs, cell_graphs.ndata['h'], with_readout=False)  # return #nodes x #classes

            # 2. backward pass
            labels = cell_graphs.ndata['labels']
            selected_labels = labels[labels != -1]
            selected_logits = logits[labels != -1, :]
            loss = loss_fn(selected_logits, selected_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 3. compute & store metrics
            mlflow.log_metric('loss', loss.item(), step=step)
            for m_name, m_fn in metrics.items():
                out = m_fn(selected_logits, selected_labels)
                mlflow.log_metric(m_name, out.item(), step=step)

            # 4. increment step
            step += 1

    #     # B.) validate
    #     model.eval()
    #     all_val_logits = []
    #     all_val_labels = []
    #     for data, labels in tqdm(dataloaders['val'], desc='Epoch validation {}'.format(epoch), unit='batch'):
    #         with torch.no_grad():
    #             labels = labels.to(DEVICE)
    #             logits = model(data)
    #         all_val_logits.append(logits)
    #         all_val_labels.append(labels)

    #     all_val_logits = torch.cat(all_val_logits).cpu()
    #     all_val_labels = torch.cat(all_val_labels).cpu()

    #     # compute & store loss + model
    #     with torch.no_grad():
    #         loss = loss_fn(all_val_logits, all_val_labels).item()
    #     mlflow.log_metric('val_loss_' + str(fold_id), loss, step=step)
    #     if loss < best_val_loss:
    #         best_val_loss = loss
    #         save_checkpoint(model, complete_path(model_path, 'model_best_val_loss_' + str(fold_id) + '.pt'))

    #     # compute & store accuracy + model
    #     accuracy = metrics['accuracy'](all_val_logits, all_val_labels).item()
    #     mlflow.log_metric('val_accuracy_' + str(fold_id), accuracy, step=step)
    #     print('Val accuracy {}'.format(accuracy))
    #     if accuracy > best_val_accuracy:
    #         best_val_accuracy = accuracy
    #         save_checkpoint(model, complete_path(model_path, 'model_best_val_accuracy_' + str(fold_id) + '.pt'))

    #     # compute & store weighted f1-score + model
    #     weighted_f1_score = metrics['weighted_f1_score'](all_val_logits, all_val_labels).item()
    #     mlflow.log_metric('val_weighted_f1_score_' + str(fold_id), weighted_f1_score, step=step)
    #     print('Weighted F1 score {}'.format(weighted_f1_score))
    #     if weighted_f1_score > best_val_weighted_f1_score:
    #         best_val_weighted_f1_score = weighted_f1_score
    #         save_checkpoint(model, complete_path(model_path, 'model_best_val_weighted_f1_score_' + str(fold_id) + '.pt'))

    #     # C) testing (at each epoch as a indication -- not used for final model prediction)
    #     all_test_logits = []
    #     all_test_labels = []
    #     for data, labels in tqdm(dataloaders['test'], desc='Testing: {}'.format(epoch), unit='batch'):
    #         with torch.no_grad():
    #             labels = labels.to(DEVICE)
    #             logits = model(data)
    #         all_test_logits.append(logits)
    #         all_test_labels.append(labels)

    #     all_test_logits = torch.cat(all_test_logits).cpu()
    #     all_test_labels = torch.cat(all_test_labels).cpu()

    #     # compute & store loss
    #     with torch.no_grad():
    #         loss = loss_fn(all_test_logits, all_test_labels).item()
    #     mlflow.log_metric('test_loss_' + str(fold_id), loss, step=step)

    #     # compute & store accuracy
    #     accuracy = metrics['accuracy'](all_test_logits, all_test_labels).item()
    #     mlflow.log_metric('test_accuracy_' + str(fold_id), accuracy, step=step)

    #     # compute & store weighted f1-score
    #     weighted_f1_score = metrics['weighted_f1_score'](all_test_logits, all_test_labels).item()
    #     mlflow.log_metric('test_weighted_f1_score_' + str(fold_id), weighted_f1_score, step=step)

    # # testing loop
    # model.eval()
    # for metric in ['best_val_loss', 'best_val_accuracy', 'best_val_weighted_f1_score']:

    #     model_name = [file for file in os.listdir(model_path) if file.endswith(".pt") and metric in file and str(fold_id) in file][0]
    #     load_checkpoint(model, complete_path(model_path, model_name))

    #     all_test_logits = []
    #     all_test_labels = []
    #     for data, labels in tqdm(dataloaders['test'], desc='Testing: {}'.format(metric), unit='batch'):
    #         with torch.no_grad():
    #             labels = labels.to(DEVICE)
    #             logits = model(data)
    #         all_test_logits.append(logits)
    #         all_test_labels.append(labels)

    #     all_test_logits = torch.cat(all_test_logits).cpu()
    #     all_test_labels = torch.cat(all_test_labels).cpu()

    #     # compute & store loss
    #     with torch.no_grad():
    #         loss = loss_fn(all_test_logits, all_test_labels).item()
    #     mlflow.log_metric('best_test_loss_' + metric + '_' + str(fold_id), loss, step=step)

    #     # compute & store accuracy
    #     accuracy = metrics['accuracy'](all_test_logits, all_test_labels).item()
    #     mlflow.log_metric('best_test_accuracy_' + metric + '_' + str(fold_id), accuracy, step=step)

    #     # compute & store weighted f1-score
    #     weighted_f1_score = metrics['weighted_f1_score'](all_test_logits, all_test_labels).item()
    #     mlflow.log_metric('best_test_weighted_f1_score_' + metric + '_' + str(fold_id), weighted_f1_score, step=step)

    #     # compute & store per class weighted f1-score 
    #     for key, val in per_class_weighted_f1_score(all_test_logits, all_test_labels, config['model_params']['class_split']).items():
    #         mlflow.log_metric('best_' + key + '_test_weighted_f1_score_' + metric + '_' + str(fold_id), val, step=step)

    #     # run external evaluators
    #     for eval_name, eval_fn in evaluators.items():

    #         out = eval_fn(all_test_logits, all_test_labels)

    #         out_path = complete_path(model_path, eval_name)
    #         out_path += DATATYPE_TO_EXT[type(out)]

    #         DATATYPE_TO_SAVEFN[type(out)](out_path, out)

    #         artifact_path = 'evaluators/{}_{}_{}'.format(eval_name, metric, fold_id)
    #         mlflow.log_artifact(out_path, artifact_path=artifact_path)

    #     # log MLflow models
    #     mlflow.pytorch.log_model(model, 'model_' + metric + '_' + str(fold_id))

    # # delete dataloaders & model
    # del dataloaders
    # del model

    # # loop over all the best metrics and compute average statistics
    # client = mlflow.tracking.MlflowClient()
    # data = client.get_run(mlflow.active_run().info.run_id).data.metrics
    # for ref in ['best_val_loss', 'best_val_accuracy', 'best_val_weighted_f1_score']:
    #     for metric in ['best_test_loss', 'best_test_accuracy', 'best_test_weighted_f1_score']:
    #         val = sum(data[metric + '_' + ref + '_' + str(id)])
    #         mlflow.log_metric('avg_' + metric.replace('best_', '') + '_' + ref, val)

    # shutil.rmtree(model_path)


if __name__ == "__main__":
    main(args=parse_arguments())
