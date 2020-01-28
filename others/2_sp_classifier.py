from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('prob_thr')               # 0:13

args = parser.parse_args()
prob_thr = float(args.prob_thr)

import numpy as np
import glob
import os
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import pickle
from PIL import Image
import cv2
import copy
from skimage.measure import regionprops
from skimage import segmentation, color
from skimage.segmentation import mark_boundaries
import time
import imageio
import h5py

def plot(img, cmap=''):
    if cmap == '':
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    plt.show()
#enddef

def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
#enddef

def prepare_data(seed=0, test_size=0.3):
    train_features = np.zeros(shape=(1, 39))
    test_features = np.zeros(shape=(1, 39))
    train_labels = []
    test_labels = []

    for t in range(len(tissue_types)):
        filepaths = glob.glob(sp_train_img_path + tissue_types[t] + '/*.png')
        filepaths.sort()

        features = np.zeros(shape=(1, 39))
        labels = []
        imgname = ''
        feats = []
        for i in range(len(filepaths)):
            basename = os.path.basename(filepaths[i]).split('.')[0]
            tumorname = basename.split('_')[1]

            chunks = basename.split('_')
            cur_imgname = chunks[0] + '_' + chunks[1] + '_' + chunks[2]
            sp_id = int(chunks[3]) - 1

            if cur_imgname != imgname:
                imgname = cur_imgname

                with h5py.File(sp_info_path + tumorname + '/' + cur_imgname + '.h5', 'r') as f:
                    data = h5py.File(sp_info_path + cur_imgname + '.h5', 'r')
                    feats = data['sp_features']

            feat = feats[sp_id, :]
            features = np.vstack((features, feat))
            labels.append(base_label[t])
        #endfor
        features = np.delete(features, 0, axis=0)

        (train_data, test_data, train_label, test_label) = train_test_split(features, labels, test_size=test_size, random_state=seed)
        train_features = np.vstack((train_features, train_data))
        test_features = np.vstack((test_features, test_data))
        train_labels += train_label
        test_labels += test_label
    #endfor
    train_features = np.delete(train_features, 0, axis=0)
    test_features = np.delete(test_features, 0, axis=0)

    # ---------------------------------------------------------------------------------------- Min-Max normalization
    min_max = np.zeros(shape=(2, train_features.shape[1]))
    for i in range(train_features.shape[1]):
        min_max[0, i] = np.min(train_features[:, i])
        min_max[1, i] = np.max(train_features[:, i])

    ## Normalize
    for i in range(train_features.shape[1]):
        minm = min_max[0, i]
        maxm = min_max[1, i]
        if maxm - minm != 0:
            train_features[:, i] = (train_features[:, i] - minm)/ (maxm - minm)
            test_features[:, i] = (test_features[:, i] - minm)/ (maxm - minm)

    if test_size == 0:
        np.savez(classifier_path + 'min_max.npz', min_max=min_max)

    return train_features, train_labels, test_features, test_labels
#enddef

# Random forest feature importance
def rf_feature_importance(train, train_label, N):
    rf = RandomForestClassifier(n_estimators= 200, criterion="gini", max_features = 0.5, n_jobs=-1)
    rf.fit(train, train_label)
    importance = rf.feature_importances_
    indices = np.argsort(importance)[::-1]
    indices = indices[:N]

    # Print the feature ranking
    '''
    print('Feature ranking:')
    for f in range(train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importance[indices[f]]))
    #'''

    # Plot feature ranking
    '''
    plt.figure()
    plt.title('Feature importances')
    plt.bar(range(train.shape[1]), importance[indices], color='r')
    plt.xticks(range(train.shape[1]), indices)
    plt.xlim([-1, train.shape[1]])
    plt.show()
    #'''

    return indices
#enddef

def svm_classifier(train, train_label, test, test_label):
    svc = SVC(kernel='rbf', C=100, gamma=0.1, random_state=0, probability=True)
    svc.fit(train, train_label)

    if test.shape[0] != 0:
        predictions = svc.predict_proba(test)
        predictions = np.argmax(predictions, axis=1)
        conf_mat = confusion_matrix(test_label, predictions)
        accuracy = sum(np.diag(conf_mat)) / float(len(test_label))
        return accuracy, conf_mat
    else:
        pickle.dump(svc, open(classifier_path + 'svm_model.pkl', 'wb'))
#enddef

def cross_validation():
    accuracy_svm = []
    n_fold = 10
    data = np.load(classifier_path + 'feature_ids.npz')
    indices = data['indices']

    for cv in range(n_fold):
        train_features, train_labels, test_features, test_labels = prepare_data(seed=cv, test_size=0.7)
        train_features = train_features[:, indices]
        test_features = test_features[:, indices]
        acc, conf_mat = svm_classifier(train=train_features, train_label=train_labels, test=test_features, test_label=test_labels)
        accuracy_svm.append(acc)
    #endfor
    accuracy_svm = np.array(accuracy_svm)
    print('\nSVM ', n_fold, 'cv accuracy: mean=', np.round(np.mean(accuracy_svm), 4), ' std=', np.round(np.std(accuracy_svm), 4))
#enddef

def feature_selection(train, train_labels, N):
    if not os.path.isfile(classifier_path + 'feature_ids.npz'):
        indices = rf_feature_importance(train=train, train_label=train_labels, N=N)
        indices = np.sort(indices)
        print('\n\nSelected ', N, ' features...')
        print(indices)
        np.savez(classifier_path + 'feature_ids.npz', indices=indices)
#enddef

def test_image(name):
    start_time = time.time()

    # ----------------------------------------------------------------------------------------------- Load image data
    img_ = Image.open(base_img_path + name + '.png')
    img = np.array(img_)
    img_.close()

    with h5py.File(sp_info_path + name + '.h5', 'r') as f:
        data = h5py.File(sp_info_path + name + '.h5', 'r')
        sp_map = data['sp_map']
        feats = data['sp_features']

    feats = feats[:, indices]
    for i in range(feats.shape[1]):
        minm = min_max[0, i]
        maxm = min_max[1, i]
        if maxm - minm != 0:
            feats[:, i] = (feats[:, i] - minm)/ (maxm - minm)
    #endfor

    # ----------------------------------------------------------------------------------------------- Predict SVM output
    pred = model.predict_proba(feats)
    pred_prob = np.max(pred, axis=1)
    pred_label = np.argmax(pred, axis=1)
    pred_label[pred_prob < prob_thr] = -1

    # ----------------------------------------------------------------------------------------------- Generate tissue map
    tissue_map = np.ones_like(sp_map) * -1
    regions = regionprops(sp_map)
    for i, region in enumerate(regions):
        if pred_label[i] != -1:
            tissue_map[sp_map == region['label']] = pred_label[i]
    #endfor

    # ----------------------------------------------------------------------------------------------- Merge super-pixels
    sp_map_new = copy.deepcopy(sp_map)

    def merge(tissue_id, map):
        mask = np.zeros_like(map)
        mask[tissue_map == tissue_id] = 255
        mask = mask.astype(np.uint8)

        num_labels, output_map, _, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_16S)
        for i in range(1, num_labels):
            id = np.unique(map[output_map == i])
            if len(id) > 1:
                map[output_map == i] = np.min(id)
        #endfor
        return map
    #enddef

    sp_map_new = merge(tissue_id=1, map=sp_map_new)
    sp_map_new = merge(tissue_id=2, map=sp_map_new)
    sp_map_new = merge(tissue_id=0, map=sp_map_new)

    print('name:', name, ' reduction:', len(np.unique(sp_map)), ':', len(np.unique(sp_map_new)), ' time:', round(time.time() - start_time, 2))

    overlaid = np.round(mark_boundaries(img, sp_map_new, (0, 0, 0)) * 255, 0).astype(np.uint8)
    instance_map = color.label2rgb(sp_map_new, img, kind='overlay')
    instance_map = np.round(segmentation.mark_boundaries(instance_map, sp_map_new, (0, 0, 0)) * 255, 0).astype(np.uint8)

    combo = np.hstack((overlaid, instance_map))
    imageio.imwrite(save_path + name + '.png', combo)
#enddef



#-----------------------------------------------------------------------------------------------------------------------
### MAIN CODE
#-----------------------------------------------------------------------------------------------------------------------
base_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/super_pixel_info/'

sp_info_path = base_path + 'basic_sp/'
sp_train_img_path = base_path + 'sp_classification/train_sp_img/'

classifier_path = base_path + 'sp_classification/sp_classifier/'
create_directory(classifier_path)

tissue_types = ['background', 'epithelium', 'necrosis', 'stroma']
base_label = [0, 1, 2, 2]

#----------------------------------------------------------------------------------------------- Select feature groups
train_features, train_labels, test_features, test_labels = prepare_data()

#----------------------------------------------------------------------------------------------- Feature selection
N = 24
feature_selection(train=train_features, train_labels=train_labels, N=N)

#----------------------------------------------------------------------------------------------- Cross-validation
cross_validation()

#----------------------------------------------------------------------------------------------- Train final model
if not os.path.isfile(classifier_path + 'svm_model.pkl'):
    train_features, train_labels, test_features, test_labels = prepare_data(test_size=0)

    data = np.load(classifier_path + 'feature_ids.npz')
    indices = data['indices']
    train_features = train_features[:, indices]

    svm_classifier(train=train_features, train_label=train_labels, test=test_features, test_label=test_labels)
    print('model saved')
#endif

exit()

#----------------------------------------------------------------------------------------------- Test final model
model = pickle.load(open(classifier_path + 'svm_model.pkl', 'rb'))
data = np.load(classifier_path + 'feature_ids.npz')
indices = data['indices']

data = np.load(classifier_path + 'min_max.npz')
min_max = data['min_max']
min_max = min_max[:, indices]

base_img_path = base_path + 'test_images/'
img_names = ['1272_udh_8', '1247_dcis_15', '1247_dcis_12', '1238_pathologicalbenign_13', '1238_pathologicalbenign_12',
             '1232_malignant_11', '1232_malignant_10', '1232_malignant_6', '1231_benign_10', '1228_pathologicalbenign_5']

save_path = base_path + 'sp_classification/test_results/'
create_directory(save_path)
save_path = save_path + 'prob_thr_' + str(prob_thr) + '/'
create_directory(save_path)

for i in range(len(img_names)):
    test_image(name=img_names[i])








