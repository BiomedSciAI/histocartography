import argparse
from config import *
import resnet

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith('__')
                     and name.startswith('resnet')
                     and callable(resnet.__dict__[name]))


parser = argparse.ArgumentParser()
parser.add_argument('--data-param', choices=['local',
                                             'dataT'],
                    default='local',
                    help='Processing location', required=False)
parser.add_argument('--mode', choices=['extract_annotations',
                                       'extract_patches',
                                       'extract_data_split',
                                       'extract_embedding',
                                       'train',
                                       'predict'],
                    default='extract_embedding',
                    help='Processing mode', required=False)

parser.add_argument('--arch', choices=model_names,
                    default='resnet32',
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('--workers',
                    default=4,
                    type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=20,
                    type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch-size',
                    default=256,
                    type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr',
                    default=0.001,
                    type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    help='momentum')
parser.add_argument('--weight-decay',
                    default=1e-4,
                    type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--dropout',
                    default=0,
                    type=float,
                    help='Dropout (default: 0)')
parser.add_argument('--pretrained',
                    default='True',
                    help='Flag to indicate using pre-trained model')
parser.add_argument('--finetune',
                    default='True',
                    help='Flag to indicate finetuning of pre-trained model')
parser.add_argument('--weighted-loss',
                    default='True',
                    help='Flag to indicate weighted cross-entropy loss')
parser.add_argument('--model-mode',
                    choices=['f1',
                             'acc',
                             'loss'],
                    default='f1',
                    help='Model mode during prediction')

args = parser.parse_args()


# Sanity check
pretrained = eval(args.pretrained)
finetune = eval(args.finetune)
if (not pretrained) and finetune:
    print('ERROR: Not supported. pretrained=False and finetune=True')
    exit()

# Instantiate configuration
config = Config(args)

# Logging parameters
print('\n\n******************************************************************************************')
print('args: ', '\n', args)
print('Patch size: ', config.patch_size)
print('Model save path: ', config.model_save_path)
print('******************************************************************************************\n\n')


if args.mode == 'extract_annotations':
    from extract_annotations import *
    extract = ExtractAnnotation(config)
    extract.get_nuclei_annotations(is_visualize=True)

elif args.mode == 'extract_patches':
    from extract_patches import *
    extract = ExtractPatches(config=config)
    extract.extract_patches(is_visualize=False)

elif args.mode == 'extract_data_split':
    from extract_data_split import *
    extract = DataSplit(config=config)
    extract.prepare_data_split()

elif args.mode == 'extract_embedding':
    from extract_embedding import *
    extract = ExtractEmbedding(config=config, args=args)
    extract.extract_embedding()

elif args.mode == 'train':
    from train import *
    train = Train(config=config, args=args)

    # Train
    train.train()

    # Test
    train.test()

elif args.mode == 'predict':
    from predict import *
    predict = Predict(config=config, args=args)
    predict.predict(is_visualize=True)




























