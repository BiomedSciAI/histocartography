from train_vae import *
from extract_patches import *
from extract_features import *
from analyze_features import *
from config import Config
import argparse
import sys
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

sys.path.append(
    '/dataT/pus/histocartography/node_embedding/histocartography/histocartography/data_generation/nuclei_features/models/')

parser = argparse.ArgumentParser()

parser.add_argument(
    '--mode',
    choices=[
        'patch_extract',
        'features_hc',
        'features_cnn',
        'features_vae'],
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
    default=48,
    help='Patch Size',
    required=False)
parser.add_argument(
    '--is_mask',
    default='False',
    help='Flag to indicate masking of nuclei patch',
    required=False)

parser.add_argument(
    '--is_train',
    default='False',
    help='Flag to indicate Training (useful for only VAE)',
    required=False)
parser.add_argument(
    '--is_patch_extraction',
    default='False',
    help='Flag to indicate Patch extraction (useful for only VAE)',
    required=False)
parser.add_argument(
    '--is_test',
    default='True',
    help='Flag to indicate feature extraction',
    required=False)
parser.add_argument(
    '--is_analysis',
    default='False',
    help='Flag to indicate feature analysis',
    required=False)

# ------------------------------------------------------------------------------------------- PRE-TRAINED CNN PARAMETERS
parser.add_argument(
    '--encoder',
    default='None',
    choices=[
        'None',
        'vgg16',
        'vgg19',
        'resnet34',
        'resnet50',
        'resnet101',
        'densenet121',
        'dennsenet169'],
    help='Pre-trained CNN encoder',
    required=False)

# ------------------------------------------------------------------------------------------- VAE TRAINING PARAMETERS
parser.add_argument(
    '--encoder_layers_per_block',
    type=int,
    default=3,
    help='Dense VAE encoder layers per block (M)',
    required=False)
parser.add_argument(
    '--embedding_dim',
    type=int,
    default=32,
    help='Size of embedding dimension (D)',
    required=False)
parser.add_argument(
    '--batch_size',
    type=int,
    default=1024,
    help='batch size',
    required=False)
parser.add_argument(
    '--num_epochs',
    type=int,
    default=200,
    help='Maximum number of training epoch',
    required=False)
parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.001,
    help='learning rate',
    required=False)
parser.add_argument(
    '--kl_weight',
    type=float,
    default=0.001,
    help='KL-divergence loss weight',
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
parser.add_argument('--tumor_type', help='Temporary tumor type', required=True)

parser.add_argument(
    '--n_chunks',
    type=int,
    default=-1,
    help='Number of chunks',
    required=False)
parser.add_argument(
    '--chunk_id',
    type=int,
    default=-1,
    help='Chunk index to be processed',
    required=False)

args = parser.parse_args()
n_chunks = args.n_chunks
chunk_id = args.chunk_id


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
    print(
        'Mode=',
        config.mode,
        ' Patch size=',
        config.patch_size,
        ' Encoder=',
        config.encoder,
        ' Experiment=',
        config.experiment_name)
    print('\n*************************************************************************************************************\n\n')

    if config.mode == 'features_hc':
        print('Extract Hand-crafted features')
        feats = Extract_HC_Features(config=config)
        feats.extract_features(chunk_id=chunk_id, n_chunks=n_chunks)

    if config.mode == 'features_cnn':
        print('Extract CNN (VGG, ResNet, DenseNet) features')

        if config.is_train:
            print('ERROR: Training of pretrained CNN is not supported.')

        if config.is_test:
            network = Extract_CNN_Features(config=config)
            feats = Extract_Deep_Features(
                config=config,
                embedding_dim=network.num_features,
                network=network)
            feats.extract_features()

        if config.is_analysis:
            network = Extract_CNN_Features(config=config)
            analyze = Analyze_Features(
                config=config, embedding_dim=network.num_features)
            analyze.feature_analysis()

    if config.mode == 'features_vae':
        if config.is_patch_extraction:
            print('Select and extract nuclei patches for training VAE')
            extract = Extract_Patches(config=config)
            extract.extract_patches(mode='train')
            extract.extract_patches(mode='val')
            # delete json: 283_benign_19, 1231_pathologicalbenign_16
            extract.extract_patches(mode='test')
            print('\n\nPatch extraction completed !!!')
            exit()
        # endif

        if config.is_train:
            eval = Patch_Evaluation(config=config)
            eval.train()
            eval.test()     # To evaluate the loss values

        if config.is_test:
            network = Extract_VAE_Features(config=config)
            feats = Extract_Deep_Features(
                config=config,
                embedding_dim=config.embedding_dim,
                network=network)
            feats.extract_features()

        if config.is_analysis:
            network = Extract_CNN_Features(config=config)
            analyze = Analyze_Features(
                config=config, embedding_dim=network.num_features)
            analyze.feature_analysis()
    # endif
