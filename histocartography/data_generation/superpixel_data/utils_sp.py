import cv2
import os
import numpy as np
import h5py
import time
from skimage.morphology import disk
from skimage.feature import greycomatrix, greycoprops
from skimage.filters.rank import entropy as Entropy
from scipy.stats import skew
from skimage.measure import regionprops
from matplotlib import pyplot as plt
from skimage.future import graph
import copy



def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
#enddef

def plot(img, cmap=''):
    if cmap == '':
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    plt.show()
#enddef

def plot_rag(img, label, rag):
    fig, ax = plt.subplots(ncols=1, sharex=True, sharey=True, figsize=(6, 8))
    lc = graph.show_rag(label, rag, img, ax=ax)
    fig.colorbar(lc, fraction=0.03, ax=ax)
    #ax.axis('off')
    plt.show()
    plt.close()
#enddef

## NOT USED
def resize(label, h, w):
    label = label.astype(np.uint8)      # conversion to uint8 assigns same id to multiple super-pixels
    label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

    if len(np.unique(label)) < 256:
        return label

    ## correcting the multiple assignment to unique assignment
    labels_resized = np.zeros(shape=(h, w))
    ctr = 1
    for j in range(len(np.unique(label))):
        mask = np.zeros_like(label)
        mask[label == j] = 255

        num_labels, output_map = cv2.connectedComponents(mask, 8, cv2.CV_32S)
        for k in range(1, num_labels):
            labels_resized[output_map == k] = ctr
            ctr += 1
        #endfor
    #endfor
    labels_resized = labels_resized.astype(int)
    return labels_resized
#enddef

def extract_basic_sp_features(img_rgb, sp_map):
    img_square = np.square(img_rgb)

    node_feat = []
    node_coord = []

    # For each super-pixel
    regions = regionprops(sp_map)

    for i, region in enumerate(regions):
        sp_mask = np.array(sp_map == region['label'], np.uint8)
        sp_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=sp_mask)
        mask_size = np.sum(sp_mask)
        mask_idx = np.where(sp_mask != 0)
        centroid = region['centroid']  # (row, column) = (y, x)

        # -------------------------------------------------------------------------- COLOR FEATURES
        ## (rgb color space) [13 x 3 features]
        def color_features_per_channel(img_rgb_ch, img_rgb_sq_ch):
            codes = img_rgb_ch[mask_idx[0], mask_idx[1]].ravel()
            hist, _ = np.histogram(codes, bins=np.arange(0, 257, 32))  # 8 bins
            feats_ = list(hist / mask_size)
            color_mean = np.mean(codes)
            color_std = np.std(codes)
            color_median = np.median(codes)
            color_skewness = skew(codes)

            codes = img_rgb_sq_ch[mask_idx[0], mask_idx[1]].ravel()
            color_energy = np.mean(codes)

            feats_.append(color_mean)
            feats_.append(color_std)
            feats_.append(color_median)
            feats_.append(color_skewness)
            feats_.append(color_energy)
            return feats_
        #enddef

        feats_r = color_features_per_channel(sp_rgb[:, :, 0], img_square[:, :, 0])
        feats_g = color_features_per_channel(sp_rgb[:, :, 1], img_square[:, :, 1])
        feats_b = color_features_per_channel(sp_rgb[:, :, 2], img_square[:, :, 2])
        feats_color = [feats_r, feats_g, feats_b]
        sp_feats = [item for sublist in feats_color for item in sublist]

        features = np.hstack(sp_feats)
        node_feat.append(features)
        node_coord.append(centroid)
    #endfor

    ## For all super-pixel in one image
    node_feat = np.vstack(node_feat)
    node_coord = np.vstack(node_coord)

    return node_feat, node_coord
#enddef

def extract_main_sp_features(img_rgb, sp_map):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_square = np.square(img_rgb)

    node_feat = []
    node_coord = []

    # -------------------------------------------------------------------------- Entropy per channel
    img_entropy = Entropy(img_gray, disk(3))

    # For each super-pixel
    regions = regionprops(sp_map)

    for i, region in enumerate(regions):
        sp_mask = np.array(sp_map == region['label'], np.uint8)
        sp_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=sp_mask)
        sp_gray = img_gray * sp_mask
        mask_size = np.sum(sp_mask)
        mask_idx = np.where(sp_mask != 0)

        # -------------------------------------------------------------------------- CONTOUR-BASED SHAPE FEATURES
        ## Compute using mask [12 features]
        centroid = region['centroid']       # (row, column) = (y, x)
        area = region['area']
        convex_area = region['convex_area']
        eccentricity = region['eccentricity']
        equivalent_diameter = region['equivalent_diameter']
        euler_number = region['euler_number']
        extent = region['extent']
        filled_area = region['filled_area']
        major_axis_length = region['major_axis_length']
        minor_axis_length = region['minor_axis_length']
        orientation = region['orientation']
        perimeter = region['perimeter']
        solidity = region['solidity']
        feats_shape = [area, convex_area, eccentricity, equivalent_diameter, euler_number, extent, filled_area,
                         major_axis_length, minor_axis_length, orientation, perimeter, solidity]

        # -------------------------------------------------------------------------- COLOR FEATURES
        ## (rgb color space) [13 x 3 features]
        def color_features_per_channel(img_rgb_ch, img_rgb_sq_ch):
            codes = img_rgb_ch[mask_idx[0], mask_idx[1]].ravel()
            hist, _ = np.histogram(codes, bins=np.arange(0, 257, 32))  # 8 bins
            feats_ = list(hist / mask_size)
            color_mean = np.mean(codes)
            color_std = np.std(codes)
            color_median = np.median(codes)
            color_skewness = skew(codes)

            codes = img_rgb_sq_ch[mask_idx[0], mask_idx[1]].ravel()
            color_energy = np.mean(codes)

            feats_.append(color_mean)
            feats_.append(color_std)
            feats_.append(color_median)
            feats_.append(color_skewness)
            feats_.append(color_energy)
            return feats_
        #enddef

        feats_r = color_features_per_channel(sp_rgb[:,:,0], img_square[:,:,0])
        feats_g = color_features_per_channel(sp_rgb[:,:,1], img_square[:,:,1])
        feats_b = color_features_per_channel(sp_rgb[:,:,2], img_square[:,:,2])
        feats_color = [feats_r, feats_g, feats_b]
        feats_color = [item for sublist in feats_color for item in sublist]

        #-------------------------------------------------------------------------- TEXTURE FEATURES
        ## Entropy (gray color space) [1 feature]
        entropy = cv2.mean(img_entropy, mask=sp_mask)[0]

        ## GLCM texture features (gray color space) [5 features]
        glcm = greycomatrix(sp_gray, [1], [0])
        filt_glcm = glcm[1:, 1:, :, :]  # Filter out the first row and column

        glcm_contrast = greycoprops(filt_glcm, prop='contrast')
        glcm_contrast = glcm_contrast[0, 0]
        glcm_dissimilarity = greycoprops(filt_glcm, prop='dissimilarity')
        glcm_dissimilarity = glcm_dissimilarity[0, 0]
        glcm_homogeneity = greycoprops(filt_glcm, prop='homogeneity')
        glcm_homogeneity = glcm_homogeneity[0, 0]
        glcm_energy = greycoprops(filt_glcm, prop='energy')
        glcm_energy = glcm_energy[0, 0]
        glcm_ASM = greycoprops(filt_glcm, prop='ASM')
        glcm_ASM = glcm_ASM[0, 0]

        feats_texture = [entropy, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_ASM]

        #-------------------------------------------------------------------------- STACKING ALL FEATURES
        sp_feats = feats_shape + feats_color + feats_texture

        features = np.hstack(sp_feats)
        node_feat.append(features)
        node_coord.append(centroid)
    #endfor

    ## For all super-pixel in one image
    node_feat = np.vstack(node_feat)
    node_coord = np.vstack(node_coord)

    return node_feat, node_coord
#enddef

def save_h5(h5_filename, sp_map, sp_feat, sp_centroid, data_dtype='float32'):
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset('sp_map', data=sp_map, dtype=data_dtype)
    h5_fout.create_dataset('sp_features', data=sp_feat, dtype=data_dtype)
    h5_fout.create_dataset('sp_centroids', data=sp_centroid, dtype=data_dtype)
    h5_fout.close()
#enddef

