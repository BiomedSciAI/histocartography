from histocartography.baseline.evaluation import (
    evaluate_troi_majority_voting,
    evaluate_troi_weighted_embedding,
    evaluate_patch
)
from histocartography.baseline.evaluation import PatchEvaluation
from histocartography.baseline.models import ModelsComponents, TwoLayerGNNCls
from histocartography.baseline.dataloader import patch_loaders, troi_loaders
from histocartography.baseline.configuration import Config
from histocartography.evaluation.classification_report import PerClassWeightedF1Score
from histocartography.evaluation.evaluator import WeightedF1
import torch.nn as nn 
import argparse
import torch
import torch.optim as optim
import mlflow 
import mlflow.pytorch
from warnings import filterwarnings
import torchvision
import numpy as np
import time
import dgl
from sklearn.metrics import f1_score
from tqdm import tqdm 
filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode',
    choices=[
        'train',
        'test',
        'merge',
        'gnn_merge'],
    default='test',
    help='Mode',
    required=False)
parser.add_argument(
    '--data_param',
    choices=[
        'local',
        'dataT'],
    default='local',
    help='Processing location',
    required=False)
parser.add_argument(
    '--split',
    type=int,
    default=3,
    help='data split index',
    required=True)
parser.add_argument(
    '--is_extraction',
    choices=[
        'True',
        'False'],
    default='False',
    help='Flag to trigger patch extraction',
    required=False)
parser.add_argument(
    '--in_ram',
    default=False,
    type=bool,
    help='Load all the data in RAM if enable',
    required=False)
parser.add_argument(
    '--model_type',
    help='Model type: options are: base, base_pt, spie',
    required=True)
parser.add_argument(
    '--is_pretrained',
    choices=[
        'True',
        'False'],
    default='True',
    help='Flag to select pre-trained models',
    required=False)

parser.add_argument(
    '--patch_size',
    type=str,
    default='10x',
    help='Patch Size',
    required=False)
parser.add_argument(
    '--patch_scale',
    type=str,
    default='10x',
    help='Patch Scale',
    required=False)
parser.add_argument(
    '--num_epochs',
    type=int,
    default=150,
    help='max epoch',
    required=False)
parser.add_argument(
    '--batch_size',
    type=int,
    default=16,
    help='batch size',
    required=False)
parser.add_argument(
    '--loss',
    default='categorical_crossentropy',
    help='loss',
    required=False)
parser.add_argument(
    '--optimizer',
    default='sgd',
    help='optimizer',
    required=False)
parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.0001,
    help='learning rate',
    required=False)
parser.add_argument(
    '--dropout',
    type=float,
    default=0.0,
    help='dropout rate',
    required=False)
parser.add_argument(
    '--weight_merge',
    default='True',
    help='Flag to indicate weighting of feature representation',
    required=False)
parser.add_argument(
    '--gpu',
    type=int,
    default=-1,
    help='gpu index',
    required=False)
parser.add_argument(
    '--class_split',
    type=str,
    default='benignVSpathologicalbenignVSudhVSadhVSfeaVSdcisVSmalignant',
    help='String defining how to split the classes. Default 7-class scenario.',
    required=False)

args = parser.parse_args()

# if args.gpu != -1:
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# ------------------------------------------------------------------------------------------- SET CONFIG
config = Config(args=args)
print('***************************************************************************************************\n')
print(
    '-split:',
    config.split,
    '-model_type:',
    config.model_type,
    '-pre_trained:',
    config.is_pretrained,
    '-patch_size:',
    config.patch_size,
    '-patch_scale:',
    config.patch_scale,
    '-batch_size:',
    config.batch_size,
    '-optimizer:',
    config.optimizer,
    '-learning_rate:',
    config.learning_rate,
    '-drop_out:',
    config.dropout)
print('***************************************************************************************************\n\n')


# -------- EXTRACT PATCHES
if config.is_extraction:
    from prepare_data import PatchExtraction
    patch_ext = PatchExtraction(config=config)

if __name__ == '__main__':
    # set the seed
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # define model
    obj = ModelsComponents(config=config)
    emb_fn = obj.embedding_model
    classifier_fn = obj.classifier_model
    is_pretrained = obj.is_pretrained
    config.is_pretrained = is_pretrained
    config.num_features = obj.num_features

    if config.mode == 'train':
        # define data loaders
        dataloaders = patch_loaders(config, is_pretrained=is_pretrained)

        # define loss function
        criterion = torch.nn.CrossEntropyLoss().to(config.device)

        # define optimizer
        if config.is_pretrained:
            params_to_update = list(classifier_fn.parameters())
        else:
            params_to_update = list(emb_fn.parameters()) + \
                list(classifier_fn.parameters())

        optimizer = optim.Adam(
            params_to_update,
            lr=config.learning_rate,
            weight_decay=0.0001)

        # start training
        eval = PatchEvaluation(config=config)
        eval.train(
            models=[
                emb_fn,
                classifier_fn],
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer)

        # -------- PATCH LEVEL EVALUATION
        evaluate_patch(
            config=config,
            mode='acc',
            eval=eval,
            dataloaders=dataloaders,
            criterion=criterion)
        evaluate_patch(
            config=config,
            mode='f1',
            eval=eval,
            dataloaders=dataloaders,
            criterion=criterion)
        evaluate_patch(
            config=config,
            mode='loss',
            eval=eval,
            dataloaders=dataloaders,
            criterion=criterion)

        # --------- TROI LEVEL EVALUATION (Majority voting)
        evaluate_troi_majority_voting(config=config, mode='acc')
        evaluate_troi_majority_voting(config=config, mode='f1')
        evaluate_troi_majority_voting(config=config, mode='loss')

    elif config.mode == 'test':
        evaluate_troi_majority_voting(config=config, mode='acc')
        evaluate_troi_majority_voting(config=config, mode='f1')
        evaluate_troi_majority_voting(config=config, mode='loss')

    elif config.mode == 'merge':
        evaluate_troi_weighted_embedding(config=config, mode='f1')

    elif config.mode == 'gnn_merge':

        mlflow.log_params({
            'class_split': args.class_split,
            'patch_size': args.patch_size
        })

        # 1. load pre-train patch classifier
        resnet34 = torchvision.models.resnet34(pretrained=True)
        emb_model = nn.Sequential(*list(resnet34.children())[:-1])

        # 2. build TRoI dataloader
        train_dataloader, val_dataloader, test_dataloader = troi_loaders(
            config, models=[emb_model, None], mode='f1')

        # 3. build GNN classifier
        gnn_model = TwoLayerGNNCls(512, config.num_classes).to(config.device)  

        # 4. training loop
        criterion = torch.nn.CrossEntropyLoss().to(config.device)
        optimizer = optim.Adam(gnn_model.parameters(), lr=config.learning_rate, weight_decay=0.0001)

        # 5. evaluators 
        eval_f1_score = WeightedF1()
        eval_per_class_f1_score = PerClassWeightedF1Score()

        best_val_f1 = -1.
        reduce_lr_count = 0
        min_reduce_lr_reached = False

        for epoch in range(config.num_epochs):
            start_time = time.time()
            print('Epoch {}/{}'.format(epoch + 1, config.num_epochs))

            # --------TRAIN MODE
            gnn_model.train()

            train_loss = 0.0
            loss = torch.FloatTensor([10e4])
            all_labels = []
            all_logits = []

            pbar = tqdm(train_dataloader, desc='Loss {}'.format(loss.item()))

            for inputs, targets in pbar: 
                optimizer.zero_grad()

                logits = gnn_model(inputs)
                loss = criterion(logits, targets)

                # measure accuracy and record loss
                all_logits.append(logits)
                all_labels.append(targets)

                train_loss += loss.cpu().detach().numpy() * len(dgl.unbatch(inputs))

                loss.backward()
                optimizer.step()
                pbar.set_description('Loss {}'.format(round(loss.item(), 3)))

            all_logits = torch.cat(all_logits).cpu()
            all_labels = torch.cat(all_labels).cpu()

            train_f1 = eval_f1_score(all_logits, all_labels)
            train_loss = train_loss / len(all_labels)

            # ------------ EVAL MODE
            gnn_model.eval()
            val_loss = 0.0
            all_labels = []
            all_logits = []

            with torch.no_grad():
                for inputs, targets in tqdm(val_dataloader):

                    logits = gnn_model(inputs)
                    loss = criterion(logits, targets)

                    all_logits.append(logits)
                    all_labels.append(targets)

                    val_loss += loss.cpu().detach().numpy() * len(dgl.unbatch(inputs))

            all_logits = torch.cat(all_logits).cpu()
            all_labels = torch.cat(all_labels).cpu()

            val_f1 = eval_f1_score(all_logits, all_labels).item()
            val_loss = val_loss / len(all_labels)

            if val_f1 > best_val_f1:
                print(
                    'val_f1 improved from', round(
                        best_val_f1, 4), 'to', round(
                        val_f1, 4), 'saving models to', config.model_save_path)
                torch.save(gnn_model, config.model_save_path + 'patch_gnn_model_best_f1.pt')
                print('Model save pathL', config.model_save_path + 'patch_gnn_model_best_f1.pt')
                best_val_f1 = val_f1

            print(' - ' + str(int(time.time() - start_time)) + 's - loss:',
                  round(train_loss,
                        4),
                  '- val_loss:',
                  round(val_loss,
                        4),
                  '- val_f1:',
                  round(val_f1,
                        4),
                  '\n')

            mlflow.log_metric('val_loss', val_loss.item(), step=epoch)
            mlflow.log_metric('val_f1', val_f1, step=epoch)

        # 5. Testing on the best model
        gnn_model = torch.load(
            config.model_save_path +
            'patch_gnn_model_best_f1.pt'
            )

        test_loss = 0.0
        all_labels = []
        all_logits = []
        with torch.no_grad():
            for inputs, targets in tqdm(test_dataloader):
                logits = gnn_model(inputs)
                loss = criterion(logits, targets)

                all_logits.append(logits)
                all_labels.append(targets)

                test_loss += loss.cpu().detach().numpy() * len(dgl.unbatch(inputs))

        all_logits = torch.cat(all_logits).cpu()
        all_labels = torch.cat(all_labels).cpu()

        # overall F1 score
        test_f1 = eval_f1_score(all_logits, all_labels).item()
        test_loss = test_loss / len(all_labels)
        print('Testing F1-score {} | Loss {}'.format(test_f1, test_loss))
        mlflow.log_metric('test_f1', test_f1)

        # per class F1 score 
        per_class_f1_score = eval_per_class_f1_score(all_logits, all_labels, args.class_split)
        print('Per class testing F1-score {}'.format(per_class_f1_score))
        for key, val in per_class_f1_score.items():
            mlflow.log_metric(key + '_test_f1', val)

        mlflow.pytorch.log_model(gnn_model, 'model_best_val_f1_score')

