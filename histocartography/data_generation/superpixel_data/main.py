from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_param')       # local, dataT
parser.add_argument('basic_flag')       # extract basic super-pixels
parser.add_argument('main_flag')        # extract main super-pixels by merging basic super-pixels
parser.add_argument('prob_thr')         # svm probability threshold for merging super-pixels
parser.add_argument('chunk_id')         # process images in chunks

args = parser.parse_args()
data_param = args.data_param
basic_flag = eval(args.basic_flag)
main_flag = eval(args.main_flag)
prob_thr = float(args.prob_thr)
chunk_id = int(args.chunk_id)

from config_sp import Config_SP
from process_sp import Process_SP

config = Config_SP(data_param=data_param, prob_thr=prob_thr)
process = Process_SP(config=config, chunk_id=chunk_id)

if basic_flag:
    process.extract_basic_superpixels(save_fig=True)

if main_flag:
    process.extract_main_superpixels(save_fig=True)

