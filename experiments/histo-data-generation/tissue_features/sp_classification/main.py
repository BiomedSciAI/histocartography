from sp_classification import *
import argparse
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('sp_type')              # basic, main
parser.add_argument('extract_feats')        # True, False

args = parser.parse_args()
sp_type = args.sp_type
extract_feats = eval(args.extract_feats)

classify = SP_Classification(sp_type=sp_type)

if extract_feats:
    classify.extract_sp_images()

if sp_type == 'basic':
    # ----------------------------------------------------------------------------------------------- Feature selection
    train, train_labels, test, test_labels = classify.prepare_data()
    classify.feature_selection(train=train, train_labels=train_labels)

    #----------------------------------------------------------------------------------------------- Cross-validation
    n_feats = 24
    classify.cross_validation(n_feats=n_feats)

    # ----------------------------------------------------------------------------------------------- Train final model
    classify.train_classifier(n_feats=n_feats)

elif sp_type == 'main':
    train, train_labels, test, test_labels = classify.prepare_data()
    classify.feature_selection(train=train, train_labels=train_labels)

    n_feats = 24
    classify.cross_validation(n_feats=n_feats)
