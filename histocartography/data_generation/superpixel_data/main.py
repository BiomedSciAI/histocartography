from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('n')
parser.add_argument('basic_flag')
parser.add_argument('main_flag')
parser.add_argument('prob_thr')

args = parser.parse_args()
n = int(args.n)
basic_flag = eval(args.basic_flag)
main_flag = eval(args.main_flag)
prob_thr = float(args.prob_thr)

from config_sp import Config_SP
from process_sp import Process_SP

config = Config_SP()
process = Process_SP(config)

if basic_flag:
    process.extract_basic_superpixels(n=n)

if main_flag:
    process.extract_main_superpixels(n=n, prob_thr=prob_thr)

