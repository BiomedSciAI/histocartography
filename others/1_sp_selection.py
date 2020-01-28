import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import imageio
import glob
import h5py
from skimage.measure import regionprops

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

def extract_sp_train():
    paths = glob.glob(sp_classifier_save_path + 'train_images/*.png')
    paths.sort()

    for i in range(len(paths)):
        basename = os.path.basename(paths[i]).split('.')[0]
        print(i, basename)
        create_directory(sp_classifier_save_path + 'train_sp_img/' + basename)

        with h5py.File(basic_sp_info_path + basename + '.h5', 'r') as f:
            data = h5py.File(basic_sp_info_path + basename + '.h5', 'r')
            sp_map = data['sp_map']
            feats = data['sp_features']

        np.random.seed(i)
        N = min(len(np.unique(sp_map)), n_sp_per_img)
        idx = np.random.choice(len(np.unique(sp_map)), N, replace=False)
        idx = np.sort(idx)

        regions = regionprops(sp_map)
        img_ = Image.open(sp_classifier_save_path + 'train_img/' + basename + '.png')
        img_rgb = np.array(img_)
        img_.close()
        (H, W, C) = img_rgb.shape

        for j, region in enumerate(regions):
            if not j in idx:
                continue

            sp_mask = np.array(sp_map == (j+1), np.uint8) * 255      # sp_map starts from 1 due to regionprops.
            min_row, min_col, max_row, max_col = region['bbox']

            min_row = 0 if (min_row - boundary < 0) else (min_row - boundary)
            max_row = H if (max_row + boundary > H) else (max_row + boundary)
            min_col = 0 if (min_col - boundary < 0) else (min_col - boundary)
            max_col = W if (max_col + boundary > W) else (max_col + boundary)

            sp_mask_crop = sp_mask[min_row:max_row, min_col:max_col]
            dilated = cv2.dilate(sp_mask_crop, kernel, iterations=2)
            sp_mask_crop = dilated - sp_mask_crop
            sp_mask_crop = 255 - sp_mask_crop

            img_rgb_crop = img_rgb[min_row:max_row, min_col:max_col, :]
            img_rgb_crop = cv2.bitwise_and(img_rgb_crop, img_rgb_crop, mask=sp_mask_crop)

            imageio.imwrite(sp_classifier_save_path + 'train_sp_img/' + basename + '/' + basename + '_' + str(j+1) + '.png', img_rgb_crop)
        #endfor
    #endfor
#enddef


#-----------------------------------------------------------------------------------------------------------------------
### MAIN CODE
#-----------------------------------------------------------------------------------------------------------------------
base_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/super_pixel_info/'

basic_sp_info_path = base_path + '1_results_basic_sp/sp_info/'
sp_classifier_save_path = base_path + 'sp_classification/'

create_directory(sp_classifier_save_path)
create_directory(sp_classifier_save_path + 'train_img/')
create_directory(sp_classifier_save_path + 'train_sp_img/')
create_directory(sp_classifier_save_path + 'sp_classifier/')

n_sp_per_img = 100
boundary = 200
kernel = np.ones((3,3),np.uint8)

extract_sp_train()
