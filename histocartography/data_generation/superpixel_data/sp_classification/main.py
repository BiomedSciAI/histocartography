from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_param')       # local, dataT

args = parser.parse_args()
data_param = args.data_param

from sp_classification import *

classify = SP_Classification(data_param=data_param)

#----------------------------------------------------------------------------------------------- Feature selection
n_feats = 24

train_features, train_labels, test_features, test_labels = classify.prepare_data()
classify.feature_selection(train=train_features, train_labels=train_labels, n_feats=n_feats)         # Random forest feature importance

#----------------------------------------------------------------------------------------------- Cross-validation
classify.cross_validation()

#----------------------------------------------------------------------------------------------- Train final model
if not os.path.isfile(classify.sp_classifier_path + 'sp_classifier/svm_model.pkl'):
    train_features, train_labels, test_features, test_labels = classify.prepare_data(test_size=0)

    data = np.load(classify.sp_classifier_path + 'sp_classifier/feature_ids.npz')
    indices = data['indices']
    train_features = train_features[:, indices]

    classify.svm_classifier(train=train_features, train_label=train_labels, test=test_features, test_label=test_labels)
    print('model saved !')
#endif
