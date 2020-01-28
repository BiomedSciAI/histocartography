import numpy as np
import os
from PIL import Image
import cv2
import imageio
import h5py
import glob
import pickle
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from skimage.measure import regionprops

class SP_Classification:
    def __init__(self, data_param):
        if data_param == 'local':
            self.basic_sp_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/super_pixel_info/basic_sp/'
            self.sp_classifier_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/misc_utils/sp_classification/'

        elif data_param == 'dataT':
            self.basic_sp_path = ''
            self.sp_classifier_path = '/dataT/pus/histocartography/Data/PASCALE/misc_utils/sp_classification/'

        self.create_directory(self.sp_classifier_path)
        self.create_directory(self.sp_classifier_path + 'train_img/')
        self.create_directory(self.sp_classifier_path + 'train_sp_img/')
        self.create_directory(self.sp_classifier_path + 'sp_classifier/')

        self.tissue_types = ['background', 'epithelium', 'necrosis', 'stroma']
        self.base_label = [0, 1, 2, 2]
        
        self.n_sp_per_img = 100
        self.boundary = 200
        
        self.n_feats = 39
    #enddef

    def create_directory(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
    #enddef

    def extract_sp_images(self):
        paths = glob.glob(self.sp_classifier_path + 'train_img/*.png')
        paths.sort()

        kernel = np.ones((3, 3), np.uint8)

        for i in range(len(paths)):
            basename = os.path.basename(paths[i]).split('.')[0]
            tumorname = basename.split('_')[1]
            print(i, basename)
            self.create_directory(self.sp_classifier_path + 'train_sp_img/' + basename)

            # --------------------------------------------------------------------------------- Load image info
            img_ = Image.open(paths[i])
            img_rgb = np.array(img_)
            img_.close()
            (H, W, C) = img_rgb.shape

            with h5py.File(self.base_sp_dir + tumorname + '/' + basename + '.h5', 'r') as f:
                sp_map = np.array(f.get('sp_map')[:]).astype(int)

            regions = regionprops(sp_map)

            np.random.seed(i)
            N = min(len(np.unique(sp_map)), self.n_sp_per_img)
            idx = np.random.choice(len(np.unique(sp_map)), N, replace=False)
            idx = np.sort(idx)

            for j, region in enumerate(regions):
                if not j in idx:
                    continue

                sp_mask = np.array(sp_map == (j + 1), np.uint8) * 255  # sp_map starts from 1 due to regionprops.
                min_row, min_col, max_row, max_col = region['bbox']

                min_row = 0 if (min_row - self.boundary < 0) else (min_row - self.boundary)
                max_row = H if (max_row + self.boundary > H) else (max_row + self.boundary)
                min_col = 0 if (min_col - self.boundary < 0) else (min_col - self.boundary)
                max_col = W if (max_col + self.boundary > W) else (max_col + self.boundary)

                sp_mask_crop = sp_mask[min_row:max_row, min_col:max_col]
                dilated = cv2.dilate(sp_mask_crop, kernel, iterations=2)
                sp_mask_crop = dilated - sp_mask_crop
                sp_mask_crop = 255 - sp_mask_crop

                img_rgb_crop = img_rgb[min_row:max_row, min_col:max_col, :]
                img_rgb_crop = cv2.bitwise_and(img_rgb_crop, img_rgb_crop, mask=sp_mask_crop)

                imageio.imwrite(self.sp_classifier_path + 'train_sp_img/' + basename + '/' + basename + '_' + str(j + 1) + '.png', img_rgb_crop)
            #endfor
        #endfor
    #enddef

    def prepare_data(self, seed=0, test_size=0.3):
        train_features = np.zeros(shape=(1, self.n_feats))
        test_features = np.zeros(shape=(1, self.n_feats))
        train_labels = []
        test_labels = []
    
        for t in range(len(self.tissue_types)):
            filepaths = glob.glob(self.sp_classifier_path + 'train_sp_img/' + self.tissue_types[t] + '/*.png')
            filepaths.sort()
    
            features = np.zeros(shape=(1, self.n_feats))
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

                    with h5py.File(self.basic_sp_path + tumorname + '/' + cur_imgname + '.h5', 'r') as f:
                        feats = np.array(f.get('sp_features')[:])


                feat = feats[sp_id, :]
                features = np.vstack((features, feat))
                labels.append(self.base_label[t])
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
            np.savez(self.sp_classifier_path + 'sp_classifier/min_max.npz', min_max=min_max)
    
        return train_features, train_labels, test_features, test_labels
    #enddef
    
    def rf_feature_importance(self, train, train_label, n_feats):
        rf = RandomForestClassifier(n_estimators= 200, criterion="gini", max_features = 0.5, n_jobs=-1)
        rf.fit(train, train_label)
        importance = rf.feature_importances_
        indices = np.argsort(importance)[::-1]
        indices = indices[:n_feats]
    
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

    def feature_selection(self, train, train_labels, n_feats):
        if not os.path.isfile(self.sp_classifier_path + 'feature_ids.npz'):
            indices = self.rf_feature_importance(train=train, train_label=train_labels, n_feats=n_feats)
            indices = np.sort(indices)
            print('\n\nSelected ', n_feats, ' features...')
            print(indices)
            np.savez(self.sp_classifier_path + 'sp_classifier/feature_ids.npz', indices=indices)
    #enddef
    
    def svm_classifier(self, train, train_label, test, test_label):
        svc = SVC(kernel='rbf', C=100, gamma=0.1, random_state=0, probability=True)
        svc.fit(train, train_label)
    
        if test.shape[0] != 0:
            predictions = svc.predict_proba(test)
            predictions = np.argmax(predictions, axis=1)
            conf_mat = confusion_matrix(test_label, predictions)
            accuracy = sum(np.diag(conf_mat)) / float(len(test_label))
            return accuracy, conf_mat
        else:
            pickle.dump(svc, open(self.sp_classifier_path + 'sp_classifier/svm_model.pkl', 'wb'))
    #enddef
    
    def cross_validation(self):
        accuracy_svm = []
        n_fold = 10
        data = np.load(self.sp_classifier_path + 'sp_classifier/feature_ids.npz')
        indices = data['indices']
    
        for cv in range(n_fold):
            train_features, train_labels, test_features, test_labels = self.prepare_data(seed=cv, test_size=0.7)
            train_features = train_features[:, indices]
            test_features = test_features[:, indices]
            acc, conf_mat = self.svm_classifier(train=train_features, train_label=train_labels, test=test_features, test_label=test_labels)
            accuracy_svm.append(acc)
        #endfor
        accuracy_svm = np.array(accuracy_svm)
        print('\nSVM ', n_fold, 'cv accuracy: mean=', np.round(np.mean(accuracy_svm), 4), ' std=', np.round(np.std(accuracy_svm), 4))
    #enddef





























    




