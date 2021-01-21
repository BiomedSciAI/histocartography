import time
import glob
from utils import *
from skimage.color import rgb2hed
import cv2 
from sklearn.metrics.pairwise import euclidean_distances
from skimage.filters.rank import entropy as Entropy
from skimage.feature import greycomatrix, greycoprops
from skimage.morphology import disk
from pathlib import Path


class ExtractFeatures:
    def __init__(self, config):
        self.config = config

    # def extract_chromatin(self, img, map):
    #     img_h = rgb2hed(img)[:, :, 0]        # Hematoxylin channel
    #     insts_list = list(np.unique(map))
    #     insts_list.remove(0)  # remove background

    #     feat_chroma = np.zeros(shape=(len(insts_list), 3))
    #     for id, inst_id in enumerate(insts_list):
    #         rows, cols = np.where(map == inst_id)
    #         pix = img_h[rows, cols]
    #         feat_chroma[id, 0] = np.mean(pix)
    #         feat_chroma[id, 1] = np.std(pix)
    #         feat_chroma[id, 2] = np.median(pix)
    #     return feat_chroma

    # def extract_roundness(self, embeddings):
    #     area = embeddings[:, 10]
    #     perimeter = embeddings[:, 13]
    #     feat_round = (4 * np.pi * area)/ (perimeter ** 2)
    #     return np.reshape(feat_round, newshape=(-1, 1))

    # def extract_ellipticity(self, embeddings):
    #     major_axis = embeddings[:, 11]
    #     minor_axis = embeddings[:, 12]
    #     feat_ellipt = minor_axis/ major_axis
    #     return np.reshape(feat_ellipt, newshape=(-1, 1))

    def bounding_box(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmax += 1
        cmax += 1
        return [rmin, rmax, cmin, cmax]

    def extract_meta_features(self, seg_map, img):

        insts_list = list(np.unique(seg_map))
        insts_list.remove(0)  # remove background
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        all_perimeters = []
        all_glcm = []

        for idx, inst_id in enumerate(insts_list):
            inst_map = np.array(seg_map == inst_id, np.uint8)
            y1, y2, x1, x2 = self.bounding_box(inst_map)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= seg_map.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= seg_map.shape[0] - 1 else y2
            nuclei_map = inst_map[y1:y2, x1:x2]

            # 1. extract convex hull perimeter
            contours, _ = cv2.findContours(nuclei_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)

            # 2. extract glcm matrix 
            nuclei_img_g = img_g[y1:y2, x1:x2]
            glcm = greycomatrix(nuclei_img_g * nuclei_map, [1], [0])
            glcm = glcm[1:, 1:, :, :]
            glcm = np.squeeze(glcm)

            all_perimeters.append(perimeter)
            all_glcm.append(glcm)

        all_perimeters = np.array(all_perimeters)
        return all_perimeters, all_glcm

    def extract_roughness(self, embeddings, cnx_h_p):
        perimeter = embeddings[:, 13]
        roughness = cnx_h_p / perimeter
        roughness = roughness[..., np.newaxis]
        return roughness

    def extract_shape_factor(self, embeddings, cnx_h_p):
        area = embeddings[:, 10]
        shape_factor = 4 * 3.1415 * area / cnx_h_p**2
        shape_factor = shape_factor[..., np.newaxis]
        return shape_factor

    def extract_std_crowdedness(self, centroids, k=10):
        dist = euclidean_distances(centroids, centroids)
        idx = np.argpartition(dist, kth=k+1, axis=-1)
        x = np.take_along_axis(dist, idx, axis=-1)[:, :k+1]
        feat_crowd = np.std(x, axis=1)
        return np.reshape(feat_crowd, newshape=(-1, 1))

    def extract_entropy(self, glcm):
        entropy = [np.mean(Entropy(g, disk(3))) for g in glcm]
        entropy = np.array(entropy)
        entropy = entropy[..., np.newaxis]
        return entropy

    def extract_dispersion(self, glcm):
        dispersion = [np.std(g) for g in glcm]
        dispersion = np.array(dispersion)
        dispersion = dispersion[..., np.newaxis]
        return dispersion

    def processing(self, img, map, embeddings, centroids):
        start_time = time.time()

        cnx_h_p, glcm = self.extract_meta_features(map, img)

        # 1. Extract roughness 
        roughness = self.extract_roughness(embeddings, cnx_h_p)

        # 2. Extract shape factor 
        shape_factor = self.extract_shape_factor(embeddings, cnx_h_p)

        # 3. Extract std of the crowdedness 
        std_crowdedness = self.extract_std_crowdedness(centroids)

        # 4. Extract entropy
        entropy = self.extract_entropy(glcm)

        # 5. extract dispersion 
        dispersion = self.extract_dispersion(glcm)

        # Stack new features to current ones 
        embeddings = np.hstack((embeddings, roughness))
        embeddings = np.hstack((embeddings, shape_factor))
        embeddings = np.hstack((embeddings, std_crowdedness))
        embeddings = np.hstack((embeddings, entropy))
        embeddings = np.hstack((embeddings, dispersion))
        print('Time= ', round(time.time() - start_time, 3))

        return embeddings

    def extract_feature(self):
        # for t in self.config.tumor_types:
        for t in ['dcis']:    

            img_paths = sorted(glob.glob(self.config.img_path + t + '/*.png'))
            print('Number of images to process: {} for tumor type {}'.format(len(img_paths), t))

            for i in range(len(img_paths)):
                if i % 10 == 0:
                    print(i, '/', len(img_paths))

                basename = os.path.basename(img_paths[i]).split('.')[0]
                instance_map_path = self.config.instance_map_path + t + '/' + basename + '.h5'
                features_path = self.config.features_path + t + '/' + basename + '.h5'
                save_path = self.config.features_path.replace('features_interpretable_', 'features_complete_') + t + '/' + basename + '.h5'
                info_path = self.config.info_path + t + '/' + basename + '.h5'

                if Path(save_path).is_file():
                    print('Save path already processed:', save_path)
                    pass
                else:

                    # ----------------------------------------------------------------- Read image information
                    img = read_image(img_paths[i])
                    map = read_instance_map(instance_map_path)
                    embeddings = read_features(features_path)
                    centroids, _ = read_info(info_path)

                    # ----------------------------------------------------------------- Feature extraction
                    embeddings = self.processing(img, map, embeddings, centroids)

                    # ----------------------------------------------------------------- Feature saving
                    h5_fout = h5py.File(save_path, 'w')
                    h5_fout.create_dataset('embeddings', data=embeddings, dtype='float32')
                    h5_fout.close()

            print('Done\n\n')





