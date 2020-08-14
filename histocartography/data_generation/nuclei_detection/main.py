from process import Process
from infer import Infer
from config import Config
import os
import argparse
import sys
from warnings import simplefilter
import logging
from absl import logging
logging._warn_preinit_stderr = 0

logging.getLogger('tensorflow').disabled = True
logging.getLogger('tensorpack').disabled = True

simplefilter(action='ignore', category=FutureWarning)

sys.path.append(
    '/dataT/pus/histocartography/node_embedding/histocartography/histocartography/data_generation/nuclei_detection/model/')
sys.path.append(
    '/dataT/pus/histocartography/node_embedding/histocartography/histocartography/data_generation/nuclei_detection/postproc/')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_param',
    choices=[
        'local',
        'dataT'],
    default='local',
    help='Processing location',
    required=False)
# benign, pathologicalbenign, udh, adh, fea, dcis, malignant
parser.add_argument(
    '--tumor_type',
    help='Tumor type to be processed',
    required=True)
parser.add_argument(
    '--gpu',
    type=int,
    default=-1,
    help='gpu index',
    required=False)
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
parser.add_argument(
    '--type_classification',
    default='False',
    help='Flag to indicate nuclei classification',
    required=False)

args = parser.parse_args()


if args.gpu != -1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


if __name__ == '__main__':
    config = Config(args=args)

    print('Inferencing...')
    infer = Infer(config=config)
    infer.run(config)

    print('\nProcessing...')
    process = Process(config=config)
    process.run()

    print('Done !')
