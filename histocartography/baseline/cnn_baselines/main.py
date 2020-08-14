from warnings import simplefilter, filterwarnings
simplefilter(action='ignore', category=FutureWarning)
filterwarnings(action='ignore', category=DeprecationWarning)

import sys

sys.path.append('/dataT/pus/histocartography/histocartography_latest/histocartography/baseline/cnn_baselines/dataloader/')
sys.path.append('/dataT/pus/histocartography/histocartography_latest/histocartography/baseline/cnn_baselines/model_single_scale/')
sys.path.append('/dataT/pus/histocartography/histocartography_latest/histocartography/baseline/cnn_baselines/model_late_fusion/')
sys.path.append('/dataT/pus/histocartography/histocartography_latest/histocartography/baseline/cnn_baselines/aggregators/')

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--mode', choices=['extract_patches',
                                       'single_scale_10x',
                                       'single_scale_20x',
                                       'single_scale_40x',
                                       'late_fusion_1020x',
                                       'late_fusion_102040x'], help='Model mode', required=True)

parser.add_argument('--encoder', choices=['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101'], help='Pre-trained CNN encoder', required=False)

parser.add_argument('--aggregator', choices=['majority_voting',
                                             'learned_fusion',
                                             'base_penultimate',
                                             'aggregate_penultimate'], help='Aggregation mode', required=False)

parser.add_argument('--class_split', default='', help='Provide class split', required=True)
parser.add_argument('--cv_split', type=int, default=1, help='Evaluation fold number', required=False)
parser.add_argument('--is_train', default='False', help='Flag to indicate training or testing', required=False)

parser.add_argument('--is_finetune', default='False', help='Flag to indicate finetuning of encoder', required=False)
parser.add_argument('--patch_size', type=int, default=112, help='Patch size at 40x', required=False)
parser.add_argument('--patch_scale', type=int, default=224, help='Input patch size for pre-trained network', required=False)

parser.add_argument('--num_epochs', type=int, default=100, help='Train number of epochs', required=False)
parser.add_argument('--batch_size', type=int, default=64, help='Batch size', required=False)
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Train learning rate', required=False)
parser.add_argument('--dropout', type=float, default=0.0, help='Drop out rate', required=False)

parser.add_argument('--info', default='', help='Additional model description', required=False)
parser.add_argument('--gpu', type=int, default=-1, help='gpu index', required=False)

args = parser.parse_args()

from config import *
import torch


if args.gpu != -1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if __name__ == '__main__':
    # set the seed
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # ------------------------------------------------------------------------------------------- SET CONFIG
    config = Config(args=args)

    print('\n*************************************************************************************************************\n')
    print('Mode=', config.mode, ' Encoder=', config.encoder, ' FineTune=', config.is_finetune, ' Class split=', config.class_split, ' CV split=', config.cv_split)
    print('\n*************************************************************************************************************\n\n')

    if config.mode == 'extract_patches':
        from extract_patches import Extract_Patches
        extract = Extract_Patches(config=config, is_balanced=True)
        extract.extract_patches()


    if config.mode in ['single_scale_10x', 'single_scale_20x', 'single_scale_40x'] :
        if config.is_train:
            from train_s import *
            eval = Train(config=config)
            eval.train()
            eval.test(modelmode='f1')

        else:
            from patch_aggregator import *
            eval = PatchAggregator(config=config)
            eval.evaluate_troi(config)


    if config.mode in ['late_fusion_1020x', 'late_fusion_102040x']:
        if config.is_train:
            from train_lf import *
            eval = Train(config=config)
            eval.train()
            eval.test(modelmode='f1')

        else:
            from patch_aggregator import *
            eval = PatchAggregator(config=config)
            eval.evaluate_troi(config)









