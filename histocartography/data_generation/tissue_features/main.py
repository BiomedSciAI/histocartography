from config import Config
import os
import argparse
import sys
from warnings import simplefilter, filterwarnings
simplefilter(action='ignore', category=FutureWarning)
filterwarnings(action='ignore', category=DeprecationWarning)

sys.path.append(
    '/dataT/pus/histocartography/node_embedding/histocartography/histocartography/data_generation/tissue_features/models/')

parser = argparse.ArgumentParser()

parser.add_argument(
    '--mode',
    choices=[
        'detect_sp',
        'features_hc',
        'features_cnn'],
    help='Mode',
    required=True)
parser.add_argument(
    '--data_param',
    choices=[
        'local',
        'dataT'],
    default='local',
    help='Processing location',
    required=False)
parser.add_argument(
    '--patch_size',
    type=int,
    default=96,
    help='Patch Size',
    required=False)
parser.add_argument(
    '--is_mask',
    default='True',
    help='Flag to indicate masking of nuclei patch',
    required=False)
parser.add_argument(
    '--merging_type',
    choices=[
        'hc',
        'cnn'],
    default='hc',
    help='Indicate the merging method',
    required=False)

# ------------------------------------------------------------------------------------------- PRE-TRAINED CNN PARAMETERS
parser.add_argument(
    '--encoder',
    choices=[
        'vgg16',
        'vgg19',
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'densenet121',
        'densenet169'],
    help='Pre-trained CNN encoder',
    required=False)

parser.add_argument(
    '--batch_size',
    type=int,
    default=512,
    help='batch size',
    required=False)
parser.add_argument(
    '--info',
    default='',
    help='Additional model description',
    required=False)
parser.add_argument(
    '--gpu',
    type=int,
    default=-1,
    help='gpu index',
    required=False)
parser.add_argument(
    '--tumor_type',
    help='Tumor type for individual processing -- temporary',
    required=True)

args = parser.parse_args()


if args.gpu != -1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------- SET CONFIG
    config = Config(args=args)

    print('\n*************************************************************************************************************\n')
    print(
        'Mode=',
        config.mode,
        ' Encoder=',
        config.encoder,
        'MergingName=',
        config.merging_name,
        ' Experiment=',
        config.features_name)
    print('\n*************************************************************************************************************\n\n')

    if config.mode == 'detect_sp':
        from extract_sp import Extract_SP
        detect = Extract_SP(config=config)
        detect.extract_sp()

    if config.mode == 'features_hc':
        from extract_hc_features import Extract_HC_Features

        features = Extract_HC_Features(config=config)
        features.extract_features()

    if config.mode == 'features_cnn':
        from extract_deep_features import Extract_Deep_Features, Extract_CNN_Features

        # set the seed
        import torch
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        network = Extract_CNN_Features(config=config)
        features = Extract_Deep_Features(
            config=config,
            embedding_dim=network.num_features,
            network=network)
        features.extract_features()
