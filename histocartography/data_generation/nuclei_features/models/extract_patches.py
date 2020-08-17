import numpy as np
import json
import glob
import h5py
import cv2
from PIL import Image
from utils import *


class Extract_Patches:
    def __init__(self, config):
        self.base_img_dir = config.base_img_dir
        self.explanation_path = config.explanation_path
        self.nuclei_detected_path = config.nuclei_detected_path

        self.vae_patches_path = config.vae_patches_path
        self.vae_masks_path = config.vae_masks_path
        self.tumor_types = config.tumor_types

        self.patch_size_2 = int(config.patch_size / 2)
        self.kernel = np.ones((3, 3), np.uint8)

        self.train_nuclei_per_tumor = [-1, 100, -1, -1, -1, 200, 200]
        self.val_nuclei_per_tumor = [-1, -1, -1, -1, -1, 500, 500]

        self.test_exclude_list = [
            '283_benign_19_explanation',
            '1231_pathologicalbenign_16_explanation']

    # enddef

    def read_image(self, path):
        img_ = Image.open(path)
        img = np.array(img_)
        (h, w, c) = img.shape
        img_.close()
        return img, h, w
    # enddef

    def read_instance_map(self, path):
        with h5py.File(path, 'r') as f:
            nuclei_instance_map = np.array(f['detected_instance_map'])
        return nuclei_instance_map
    # enddef

    def read_nuclei_loc(self, path):
        with open(path) as f:
            data = json.load(f)
            centroids_ = data['output']['explanation']['centroids']
            centroids_ = centroids_.split('], [')
            centroids_[0] = centroids_[0].split('[[')[1]
            centroids_[-1] = centroids_[-1].split(']]')[0]

            centroids = []
            for j in range(len(centroids_)):
                x = centroids_[j].split(', ')
                centroids.append([int(float(a)) for a in x])
        # end
        return centroids
    # enddef

    def get_filenames(self, path):
        basename = os.path.basename(path).split('.')[0]
        chunks = basename.split('_')
        tumor_type = chunks[1]
        tumor_index = self.tumor_types.index(tumor_type)

        filename = chunks[0] + '_' + chunks[1] + '_' + chunks[2]
        img_path = self.base_img_dir + tumor_type + '/' + filename + '.png'
        nuclei_detected_path = self.nuclei_detected_path + \
            tumor_type + '/_h5/' + filename + '.h5'

        return basename, filename, tumor_type, tumor_index, img_path, nuclei_detected_path
    # enddef

    def get_centroid_index(self, mode, tumor_index, n_centroid, seed):
        if mode == 'train':
            N = self.train_nuclei_per_tumor[tumor_index]
        elif mode == 'val':
            N = self.val_nuclei_per_tumor[tumor_index]

        if N == -1:
            idx = np.arange(n_centroid)
        else:
            np.random.seed(seed)
            N = min(N, n_centroid)
            idx = np.random.choice(np.arange(n_centroid), N, replace=False)

        return idx
    # enddef

    def extract_patches(self, mode):
        print('Extracting nuclei for: ', mode)
        create_directory(self.vae_patches_path + mode)
        create_directory(self.vae_masks_path + mode)

        if mode != 'test':
            explanation_paths = glob.glob(
                self.explanation_path + mode + '/fold_4/*.json')
        else:
            explanation_paths = glob.glob(
                self.explanation_path + mode + '/fold_4/*explanation.json')

        tumor_nuclei_count = [0 for i in range(len(self.tumor_types))]
        tumor_count = [0 for i in range(len(self.tumor_types))]

        for i in range(len(explanation_paths)):
            basename, filename, tumor_type, tumor_index, img_path, nuclei_info_path = self.get_filenames(
                explanation_paths[i])
            if basename in self.test_exclude_list:
                continue

            if os.path.isfile(img_path) and os.path.isfile(nuclei_info_path):
                # ----------------------------------------------------------------- Read nuclei locations
                centroids = self.read_nuclei_loc(explanation_paths[i])

                # ----------------------------------------------------------------- Extract nuclei_patch and nuclei_mask for each nuclei
                img, h, w = self.read_image(img_path)
                nuclei_instance_map = self.read_instance_map(nuclei_info_path)
                idx = self.get_centroid_index(
                    mode, tumor_index, len(centroids), seed=i)

                nuclei_count = 0
                boundary_nuclei_count = 0

                for j in range(len(idx)):
                    x = centroids[idx[j]][0]
                    y = centroids[idx[j]][1]

                    if (y -
                        self.patch_size_2 > 0) and (y +
                                                    self.patch_size_2 < h) and (x -
                                                                                self.patch_size_2 > 0) and (x +
                                                                                                            self.patch_size_2 < w):
                        nuclei_count += 1
                        patch = img[y -
                                    self.patch_size_2: y +
                                    self.patch_size_2, x -
                                    self.patch_size_2: x +
                                    self.patch_size_2, :]
                        Image.fromarray(patch).save(
                            self.vae_patches_path + mode + '/' + basename + '_' + str(j) + '.png')

                        mask_ = nuclei_instance_map[y -
                                                    self.patch_size_2: y +
                                                    self.patch_size_2, x -
                                                    self.patch_size_2: x +
                                                    self.patch_size_2]
                        mask = np.zeros_like(mask_)
                        mask[mask_ == idx[j] + 1] = 255
                        mask[mask_ != idx[j] + 1] = 0
                        mask = mask.astype(np.uint8)
                        mask = cv2.dilate(mask, self.kernel, iterations=1)
                        Image.fromarray(mask).save(
                            self.vae_masks_path + mode + '/' + basename + '_' + str(j) + '.png')
                    else:
                        boundary_nuclei_count += 1
                # endfor

                tumor_nuclei_count[tumor_index] += nuclei_count
                tumor_count[tumor_index] += 1

                print(
                    i,
                    'File name=',
                    filename,
                    ' #nuclei=',
                    nuclei_count,
                    ' #boundary_nuclei=',
                    boundary_nuclei_count)
            # endif
        # endfor

        print('\nTUMOR-WISE COUNT:\n')
        for i in range(len(self.tumor_types)):
            print(
                self.tumor_types[i],
                ': #troi=',
                tumor_count[i],
                ' #nuclei',
                tumor_nuclei_count[i])
    # enddef
# end
