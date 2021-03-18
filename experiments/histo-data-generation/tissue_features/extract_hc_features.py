import glob
import h5py
from PIL import Image
import cv2
from skimage.feature import greycomatrix, greycoprops
from skimage.filters.rank import entropy as Entropy
from skimage.morphology import disk
from skimage.measure import regionprops
from scipy.stats import skew
from utils import *
import numpy as np


class Extract_HC_Features:
    def __init__(self, config):
        self.base_img_dir = config.base_img_dir
        self.mode = config.mode
        self.tumor_types = config.tumor_types

        self.sp_detected_path = config.sp_merged_detected_path
        self.sp_features_path = config.sp_merged_features_path

        self.tumor_type = config.tumor_type
    # enddef

    def read_image(self, path):
        img_ = Image.open(path)
        img = np.array(img_)
        (h, w, c) = img.shape
        img_.close()
        return img, h, w
    # enddef

    def read_instance_map(self, map_path):
        with h5py.File(map_path, 'r') as f:
            sp_instance_map = np.array(f['detected_instance_map']).astype(int)

        return sp_instance_map
    # enddef

    def bounding_box(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmax += 1
        cmax += 1
        return [rmin, rmax, cmin, cmax]
    # enddef

    def processing(self, img_rgb, sp_map):
        node_feat = []

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_square = np.square(img_rgb)

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
            # Compute using mask [12 features]
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
            feats_shape = [
                area,
                convex_area,
                eccentricity,
                equivalent_diameter,
                euler_number,
                extent,
                filled_area,
                major_axis_length,
                minor_axis_length,
                orientation,
                perimeter,
                solidity]

            # -------------------------------------------------------------------------- COLOR FEATURES
            # (rgb color space) [13 x 3 features]
            def color_features_per_channel(img_rgb_ch, img_rgb_sq_ch):
                codes = img_rgb_ch[mask_idx[0], mask_idx[1]].ravel()
                hist, _ = np.histogram(
                    codes, bins=np.arange(
                        0, 257, 32))  # 8 bins
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
            # enddef

            feats_r = color_features_per_channel(
                sp_rgb[:, :, 0], img_square[:, :, 0])
            feats_g = color_features_per_channel(
                sp_rgb[:, :, 1], img_square[:, :, 1])
            feats_b = color_features_per_channel(
                sp_rgb[:, :, 2], img_square[:, :, 2])
            feats_color = [feats_r, feats_g, feats_b]
            feats_color = [item for sublist in feats_color for item in sublist]

            # -------------------------------------------------------------------------- TEXTURE FEATURES
            # Entropy (gray color space) [1 feature]
            entropy = cv2.mean(img_entropy, mask=sp_mask)[0]

            # GLCM texture features (gray color space) [5 features]
            glcm = greycomatrix(sp_gray, [1], [0])
            # Filter out the first row and column
            filt_glcm = glcm[1:, 1:, :, :]

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

            feats_texture = [
                entropy,
                glcm_contrast,
                glcm_dissimilarity,
                glcm_homogeneity,
                glcm_energy,
                glcm_ASM]

            # -------------------------------------------------------------------------- STACKING ALL FEATURES
            sp_feats = feats_shape + feats_color + feats_texture

            features = np.hstack(sp_feats)
            node_feat.append(features)
        # endfor

        node_feat = np.vstack(node_feat)
        return node_feat
    # endfor

    def extract_features(self):
        # for tumor_type in self.tumor_types:
        tumor_type = self.tumor_type

        print('Extracting Hand-crafted features for: ', tumor_type)
        img_filepaths = sorted(
            glob.glob(
                self.base_img_dir +
                tumor_type +
                '/*.png'))

        for i in range(len(img_filepaths)):
            if i % 20 == 0:
                print(i, '/', len(img_filepaths))

            basename = os.path.basename(img_filepaths[i]).split('.')[0]
            sp_map_path = self.sp_detected_path + 'instance_map/' + \
                tumor_type + '/' + basename + '.h5'

            if os.path.isfile(
                self.sp_features_path +
                tumor_type +
                '/' +
                basename +
                    '.h5'):
                with h5py.File(self.sp_features_path + tumor_type + '/' + basename + '.h5', 'r') as f:
                    embeddings = np.array(f['embeddings'])
                # continue

            if os.path.isfile(sp_map_path):
                # ----------------------------------------------------------------- Read image information
                # SP instance map
                map = self.read_instance_map(sp_map_path)

                if len(np.unique(map)) != embeddings.shape[0]:
                    # Image
                    img, h, w = self.read_image(img_filepaths[i])

                    centroid_path = self.sp_detected_path + \
                        'centroids/' + tumor_type + '/' + basename + '.h5'
                    with h5py.File(centroid_path, 'r') as f:
                        centroids = np.array(f['instance_centroid_location'])

                    print(basename,
                          ':',
                          centroids.shape[0],
                          ':',
                          len(np.unique(map)),
                          ':',
                          embeddings.shape[0],
                          ':',
                          img.shape,
                          ':',
                          map.shape)

                    # ----------------------------------------------------------------- Feature extraction
                    embeddings = self.processing(img, map)

                    # ----------------------------------------------------------------- Feature saving
                    h5_fout = h5py.File(
                        self.sp_features_path +
                        tumor_type +
                        '/' +
                        basename +
                        '.h5',
                        'w')
                    h5_fout.create_dataset(
                        'embeddings', data=embeddings, dtype='float32')
                    h5_fout.close()
            # endif
        # endfor
        print('Done\n\n')
        # endfor
    # enddef
# end
