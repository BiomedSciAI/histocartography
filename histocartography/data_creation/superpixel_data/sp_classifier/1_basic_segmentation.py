from sklearn.cross_validation import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
import os
import numpy as np
import random
import cv2
import imageio
from PIL import Image
import glob
from pixel_selection import PixelSelection
random.seed(0)

##----------------------------------------------------------------------------------------------------------------------
### SUPPORTING FUNCTIONS
##----------------------------------------------------------------------------------------------------------------------
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

## Create masks for Hematoxylin, Eosin and Background
def get_B_and_rest_mask(OD, OD_R, OD_G, OD_B):
    # Extract pixel locations that are not Background
    OD_loc = (OD < 0.2)
    OD_R_loc = (OD_R < 0.25)
    OD_G_loc = (OD_G < 0.25)
    OD_B_loc = (OD_B < 0.25)

    ################################################# Background pixel mask
    mask_B = np.bitwise_and(OD_loc, OD_R_loc)
    mask_B = np.bitwise_and(mask_B, OD_G_loc)
    mask_B = np.bitwise_and(mask_B, OD_B_loc)

    # OD_loc = (OD >= 0.2)
    # OD_R_loc = (OD_R >= 0.1)
    # OD_G_loc = (OD_G >= 0.25)
    # OD_B_loc = (OD_B >= 0.25)
    # loc = np.bitwise_and(OD_loc, OD_R_loc)
    # loc = np.bitwise_and(loc, OD_G_loc)
    # loc = np.bitwise_and(loc, OD_B_loc)
    # mask_H_E = OD_loc
    # mask_B = np.logical_not(mask_H_E)

    mask_H_E = np.logical_not(mask_B)

    return mask_B, mask_H_E
#enddef

## Scatter plot of (Cx, Cy) distributions of H, E and B class
def scatter_plot_2d(pixel_info_B, pixel_info_H, pixel_info_E):
    ## Scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n = 1000
    ax.scatter(pixel_info_B[0:n, 0], pixel_info_B[0:n, 1], c='g', marker='o')
    ax.scatter(pixel_info_H[0:n,0], pixel_info_H[0:n,1], c='b', marker='o')
    ax.scatter(pixel_info_E[0:n,0], pixel_info_E[0:n,1], c='r', marker='o')
    ax.set_xlabel('Cx')
    ax.set_ylabel('Cy')
    #plt.show()
#enddef

## Scatter plot of (Cx, Cy, D) distributions of H, E and B class
def scatter_plot_3d(pixel_info_B, pixel_info_H, pixel_info_E):
    ## Scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 1000
    ax.scatter(pixel_info_B[0:n,0], pixel_info_B[0:n,1], pixel_info_B[0:n,2], c='g', marker='o')
    ax.scatter(pixel_info_H[0:n,0], pixel_info_H[0:n,1], pixel_info_H[0:n,2], c='b', marker='o')
    ax.scatter(pixel_info_E[0:n,0], pixel_info_E[0:n,1], pixel_info_E[0:n,2], c='r', marker='o')
    ax.set_xlabel('Cx')
    ax.set_ylabel('Cy')
    ax.set_zlabel('Density')
    ax.view_init(elev=40., azim=-130)
    plt.show()
#enddef

def majority_voting(arr):
    arr = arr.astype(int)
    counts = np.bincount(arr)
    return np.argmax(counts)
#enddef


##----------------------------------------------------------------------------------------------------------------------
### TRAINING A CLASSIFIER FOR SEGMENTATION
##----------------------------------------------------------------------------------------------------------------------
def prepare_segmentation_data():
    pixelselection = PixelSelection()

    for i in range(len(img_names)):
        n = 1
        if (i < 5 * n) or (i > 5 * (n + 1)):
            continue

        base_name = os.path.basename(img_names[i])
        if not os.path.isfile(pixel_classification_path + 'pixel_data/' + base_name + '.npz'):
            print('#', i, ' : ', img_names[i], '...')
            path = img_path + tumor_types[i] + '/' + img_names[i]
            b, h, e, figure = pixelselection.extract_pixel_samples(path)
            imageio.imwrite(pixel_images_path + img_names[i], figure)
            np.savez(pixel_data_path + base_name + '.npz', B_pixels=b, H_pixels=h, E_pixels=e)
        #endfor
    #endif
#enddef

## Load data for individual image
def load_data_for_segmentation():
    H_pixels = np.zeros(shape=(1, 3))
    E_pixels = np.zeros(shape=(1, 3))
    for i in range(len(img_names)):
        base_name = os.path.basename(img_names[i])
        data = np.load(pixel_data_path + base_name + '.npz')

        H_pixels = np.vstack((H_pixels, data['H_pixels']))
        E_pixels = np.vstack((E_pixels, data['E_pixels']))
    #endfor
    H_pixels = np.delete(H_pixels, 0, axis=0)
    E_pixels = np.delete(E_pixels, 0, axis=0)
    return H_pixels, E_pixels
#endfor

## Load and create training and validation data for training the neural network
def create_train_validation_data_for_segmentation():
    H_pixels, E_pixels = load_data_for_segmentation()
    print('H_pixels:', H_pixels.shape, 'E_pixels:', E_pixels.shape)

    # Randomly select points from H and E class for training
    np.random.seed(0)
    index = np.random.choice(H_pixels.shape[0], n_pixels_per_class, replace=False)
    H_pixels = H_pixels[index, :]

    np.random.seed(1)
    index = np.random.choice(E_pixels.shape[0], n_pixels_per_class, replace=False)
    E_pixels = E_pixels[index, :]

    ## Min_Max values for individual features on complete data
    data = np.vstack((H_pixels, E_pixels))
    min_max_norms = np.full((2, data.shape[1]), 0)
    for i in range(data.shape[1]):
        min_max_norms[0, i] = np.min(data[:, i])
        min_max_norms[1, i] = np.max(data[:, i])

        minm = min_max_norms[0, i]
        maxm = min_max_norms[1, i]
        if maxm - minm != 0:
            data[:, i] = (data[:, i] - minm) / (maxm - minm)
    #endfor
    np.savez(pixel_classification_path + 'min_max_norms.npz', min_max_norms = min_max_norms)

    H_labels = np.full(H_pixels.shape[0], 0)
    E_labels = np.full(E_pixels.shape[0], 1)
    labels = np.append(H_labels, E_labels)

    labels_categorical = np.eye(n_segmentation_classes)[labels]
    print('data: ', data.shape, ', labels: ', labels_categorical.shape, ', labels per class:', np.sum(labels_categorical, axis=0))
    np.savez(pixel_classifier_path + 'train_and_validation.npz', data=data, labels=labels, labels_categorical=labels_categorical)

    return data, labels, labels_categorical
#enddef

## Train the segmentation classifier
def train_segmentation_model():
    print('Training segmentation classifier...')

    if not os.path.isfile(pixel_classifier_path + 'train_and_validation.npz'):
        data, labels, labels_categorical = create_train_validation_data_for_segmentation()
    else:
        train_and_validation = np.load(pixel_classifier_path + 'train_and_validation.npz')
        data = train_and_validation['data']
        labels = train_and_validation['labels']
        labels_categorical = train_and_validation['labels_categorical']
    #endif

    ## Train SVM
    (train_data, validation_data, train_labels, validation_labels) = train_test_split(data, labels, test_size=0.3, random_state=0)
    ## SVM
    # clf = SVC(random_state=0, kernel='rbf')
    # param_grid = {'C': [100, 1000, 10000], 'gamma': [1, 10, 100]}
    # clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3)
    # clf.fit(train_data, train_labels)
    #svc = SVC(kernel='rbf', C=clf.best_params_['C'], gamma=clf.best_params_['gamma'], random_state=0, probability=True)

    svc = SVC(kernel='rbf', C=10000, gamma=10, random_state=0, probability=True)
    svc.fit(train_data, train_labels)

    predictions = svc.predict_proba(validation_data)
    predictions = np.argmax(predictions, axis=1)

    conf_mat = confusion_matrix(validation_labels, predictions)
    accuracy = sum(np.diag(conf_mat)) / float(len(validation_labels))
    print('SVM accuracy: ' + str(round(accuracy, 4)))
    print(conf_mat)

    pickle.dump(svc, open(save_svm_model_path, 'wb'))
#enddef


##----------------------------------------------------------------------------------------------------------------------
### SEGMENTING A TEST IMAGE
##----------------------------------------------------------------------------------------------------------------------
## Segment H, E and B categories in a test tile image using the trained neural network
def test_segmentation(path, imagename):
    img_rgb = np.array(Image.open(path))

    OD, OD_R, OD_G, OD_B = pixelselection.get_density_components(img_rgb)
    mask_B, mask_H_E = get_B_and_rest_mask(OD, OD_R, OD_G, OD_B)

    ## Extract features (Cx, Cy, D) for each pixel
    pixel_info, pts = pixelselection.compute_pixel_sample_features(mask_H_E, OD, OD_R, OD_G, OD_B)

    ## Normalize test data
    min_max_norms = np.load(pixel_classification_path + 'min_max_norms.npz')
    min_max_norms = min_max_norms['min_max_norms']
    for i in range(pixel_info.shape[1]):
        minm = min_max_norms[0, i]
        maxm = min_max_norms[1, i]
        if maxm - minm != 0:
            pixel_info[:, i] = (pixel_info[:, i] - minm)/ (maxm - minm)
        #endif
    #endfor

    ## Load trained model, and predict H and E label for all pixels
    trained_model = pickle.load(open(save_svm_model_path, 'rb'))
    predictions = trained_model.predict_proba(pixel_info)
    predictions = np.argmax(predictions, axis=1)
    pts = np.asarray(pts)
    pts = np.transpose(pts)

    index_B = np.transpose(np.asarray(np.where(mask_B == True)))
    index_H = pts[np.where(predictions == 0)[0], :]
    index_E = pts[np.where(predictions == 1)[0], :]

    ## Plot the predictions
    predicted_image = np.zeros_like(img_rgb)
    predicted_image[index_H[:,0], index_H[:,1], :] = (0, 102, 255)
    predicted_image[index_E[:,0], index_E[:,1], :] = (255, 102, 255)
    predicted_image[index_B[:,0], index_B[:,1], :] = (255, 255, 255)
    combined = np.hstack((img_rgb, predicted_image))
    imageio.imwrite(pixel_test_images_path + imagename, combined)
#edndef


def get_image_file_paths():
    file_paths_all = []
    for i in range(len(tumor_types)):
        file_paths = glob.glob(img_path + '*.png')

        ## Only useful for sp_classification
        np.random.seed(0)
        idx = np.random.choice(len(file_paths), 20)
        file_paths = [file_paths[i] for i in idx]
        file_paths.sort()

        file_paths_all += file_paths
    #endfor
    return file_paths_all
#enddef

###################################################################################################################################
### MAIN CODE
###################################################################################################################################
threshold = 30              # 10, 20

tumor_types = ['0_benign', '1_pathological_benign', '5_dcis', '6_malignant']

img_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/Images_norm/'
sp_feat_select_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/super_pixel_info/sp_feat_selection/'
create_directory(sp_feat_select_path)

pixel_classification_path = sp_feat_select_path + 'pixel_classification/'
create_directory(pixel_classification_path)

pixel_data_path = pixel_classification_path + 'pixel_data/'
pixel_images_path = pixel_classification_path + 'pixel_images/'
pixel_classifier_path = pixel_classification_path + 'pixel_classifier/'
pixel_test_images_path = pixel_classification_path + 'test_images/'
create_directory(pixel_data_path)
create_directory(pixel_images_path)
create_directory(pixel_classifier_path)
create_directory(pixel_test_images_path)


n_segmentation_classes = 2
n_pixels_per_class = 100000
save_svm_model_path = pixel_classifier_path + 'segmentation_svm.pkl'


if __name__ == "__main__":
    prepare_segmentation_data()

    if not os.path.isfile(save_svm_model_path):
        train_segmentation_model()

    ## Test an image
    tumor_types = ['0_benign', '1_pathological_benign', '5_dcis', '6_malignant']
    img_names = ['295_benign_4.png', '743_pathologicalbenign_2.png', '773_dcis_15.png', '291_malignant_8.png']

    pixelselection = PixelSelection()

    for i in range(len(img_names)):
        path = img_path + tumor_types[i] + '/' + img_names[i]
        test_segmentation(path, img_names[i])
#endif














