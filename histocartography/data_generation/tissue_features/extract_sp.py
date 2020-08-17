from PIL import Image
import glob
import cv2
import h5py
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.future.graph import RAG
from skimage.future import graph
import pickle
from scipy.stats import skew
import copy
from utils import *


class Extract_SP:
    def __init__(self, config):
        self.tumor_types = config.tumor_types
        self.base_img_dir = config.base_img_dir
        self.sp_unmerged_detected_path = config.sp_unmerged_detected_path
        self.sp_merged_detected_path = config.sp_merged_detected_path
        self.sp_classifier_path = config.sp_classifier_path

        self.base_n_segments = config.base_n_segments
        self.base_n_pixels = config.base_n_pixels
        self.max_n_segments = config.max_n_segments

        self.magnification = 8  # magnification level
        self.blur_kernel_size = 3  # blurring kernel size
        self.threshold = 0.01
        self.w_hist = 0.5
        self.w_mean = 0.5
        self.prob_thr = 0.8

        self.tumor_type = config.tumor_type
    # enddef

    def read_image(self, path):
        img_ = Image.open(path)
        img = np.array(img_)
        (h, w, c) = img.shape
        img_.close()
        return img, h, w
    # enddef

    def color_features_per_channel(self, img_ch):
        hist, _ = np.histogram(img_ch, bins=np.arange(0, 257, 64))  # 8 bins
        return hist
    # enddef

    def weight_mean_color(self, graph, src, dst, n):
        diff_mean = np.linalg.norm(
            graph.nodes[dst]['mean'] -
            graph.nodes[n]['mean'])

        diff_r = np.linalg.norm(
            graph.nodes[dst]['r'] - graph.nodes[n]['r']) / 2
        diff_g = np.linalg.norm(
            graph.nodes[dst]['g'] - graph.nodes[n]['g']) / 2
        diff_b = np.linalg.norm(
            graph.nodes[dst]['b'] - graph.nodes[n]['b']) / 2
        diff_hist = (diff_r + diff_g + diff_b) / 3

        diff = self.w_hist * diff_hist + self.w_mean * diff_mean

        return {'weight': diff}
    # enddef

    def merge_mean_color(self, graph, src, dst):
        graph.nodes[dst]['x'] += graph.nodes[src]['x']
        graph.nodes[dst]['N'] += graph.nodes[src]['N']
        graph.nodes[dst]['mean'] = (
            graph.nodes[dst]['x'] /
            graph.nodes[dst]['N'])
        graph.nodes[dst]['mean'] = graph.nodes[dst]['mean'] / \
            np.linalg.norm(graph.nodes[dst]['mean'])

        graph.nodes[dst]['y'] = np.vstack(
            (graph.nodes[dst]['y'], graph.nodes[src]['y']))
        graph.nodes[dst]['r'] = self.color_features_per_channel(
            graph.nodes[dst]['y'][:, 0])
        graph.nodes[dst]['g'] = self.color_features_per_channel(
            graph.nodes[dst]['y'][:, 1])
        graph.nodes[dst]['b'] = self.color_features_per_channel(
            graph.nodes[dst]['y'][:, 2])

        graph.nodes[dst]['r'] = graph.nodes[dst]['r'] / \
            np.linalg.norm(graph.nodes[dst]['r'])
        graph.nodes[dst]['g'] = graph.nodes[dst]['r'] / \
            np.linalg.norm(graph.nodes[dst]['g'])
        graph.nodes[dst]['b'] = graph.nodes[dst]['r'] / \
            np.linalg.norm(graph.nodes[dst]['b'])
    # enddef

    def rag_mean_std_color(self, image, labels, connectivity=2):
        graph = RAG(labels, connectivity=connectivity)

        for n in graph:
            graph.nodes[n].update({'labels': [n],
                                   'N': 0,
                                   'x': np.array([0, 0, 0]),
                                   'y': np.array([0, 0, 0]),
                                   'r': np.array([]),
                                   'g': np.array([]),
                                   'b': np.array([])})

        for index in np.ndindex(labels.shape):
            current = labels[index]
            graph.nodes[current]['N'] += 1
            graph.nodes[current]['x'] += image[index]
            graph.nodes[current]['y'] = np.vstack(
                (graph.nodes[current]['y'], image[index]))

        for n in graph:
            graph.nodes[n]['mean'] = (
                graph.nodes[n]['x'] / graph.nodes[n]['N'])
            graph.nodes[n]['mean'] = graph.nodes[n]['mean'] / \
                np.linalg.norm(graph.nodes[n]['mean'])

            graph.nodes[n]['y'] = np.delete(graph.nodes[n]['y'], 0, axis=0)
            graph.nodes[n]['r'] = self.color_features_per_channel(
                graph.nodes[n]['y'][:, 0])
            graph.nodes[n]['g'] = self.color_features_per_channel(
                graph.nodes[n]['y'][:, 1])
            graph.nodes[n]['b'] = self.color_features_per_channel(
                graph.nodes[n]['y'][:, 2])

            graph.nodes[n]['r'] = graph.nodes[n]['r'] / \
                np.linalg.norm(graph.nodes[n]['r'])
            graph.nodes[n]['g'] = graph.nodes[n]['r'] / \
                np.linalg.norm(graph.nodes[n]['g'])
            graph.nodes[n]['b'] = graph.nodes[n]['r'] / \
                np.linalg.norm(graph.nodes[n]['b'])

        for x, y, d in graph.edges(data=True):
            diff_mean = np.linalg.norm(
                graph.nodes[x]['mean'] - graph.nodes[y]['mean']) / 2

            diff_r = np.linalg.norm(
                graph.nodes[x]['r'] - graph.nodes[y]['r']) / 2
            diff_g = np.linalg.norm(
                graph.nodes[x]['g'] - graph.nodes[y]['g']) / 2
            diff_b = np.linalg.norm(
                graph.nodes[x]['b'] - graph.nodes[y]['b']) / 2
            diff_hist = (diff_r + diff_g + diff_b) / 3

            diff = self.w_hist * diff_hist + self.w_mean * diff_mean

            d['weight'] = diff

        return graph
    # enddef

    def extract_sp_classification_features(self, img_rgb, sp_map):
        img_square = np.square(img_rgb)

        node_feat = []
        regions = regionprops(sp_map)

        for i, region in enumerate(regions):
            sp_mask = np.array(sp_map == region['label'], np.uint8)
            sp_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=sp_mask)
            mask_size = np.sum(sp_mask)
            mask_idx = np.where(sp_mask != 0)

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
            sp_feats = [item for sublist in feats_color for item in sublist]

            features = np.hstack(sp_feats)
            node_feat.append(features)
        # endfor

        node_feat = np.vstack(node_feat)
        return node_feat
    # enddef

    def load_sp_classifier(self):
        self.svm = pickle.load(
            open(
                self.sp_classifier_path +
                'svm_model.pkl',
                'rb'))
        # self.svm.__getstate__()['_sklearn_version']

        data = np.load(self.sp_classifier_path + 'feature_ids.npz')
        self.indices = data['indices']
        self.indices = np.sort(self.indices)

        data = np.load(self.sp_classifier_path + 'min_max.npz')
        self.min_max = data['min_max']
        self.min_max = self.min_max[:, self.indices]
    # enddef

    def classification_merge(self, features, sp_map):
        # ---------------------------------------------------------------------------------------------- Select features
        features = features[:, self.indices]

        # ---------------------------------------------------------------------------------------------- Normalize
        for j in range(features.shape[1]):
            minm = self.min_max[0, j]
            maxm = self.min_max[1, j]
            if maxm - minm != 0:
                features[:, j] = (features[:, j] - minm) / (maxm - minm)
        # endfor

        # ---------------------------------------------------------------------------------------------- Select features
        pred = self.svm.predict_proba(features)
        pred_prob = np.max(pred, axis=1)
        pred_label = np.argmax(pred, axis=1)
        pred_label[pred_prob < self.prob_thr] = -1

        # ---------------------------------------------------------------------------------------------- Generate tissue map
        tissue_map = np.ones_like(sp_map) * -1
        regions = regionprops(sp_map)
        for j, region in enumerate(regions):
            if pred_label[j] != -1:
                tissue_map[sp_map == region['label']] = pred_label[j]
        # endfor

        # ----------------------------------------------------------------------------------------------- Merge super-pixels
        sp_map_merged = copy.deepcopy(sp_map)

        def merge(tissue_id, map):
            mask = np.zeros_like(map)
            mask[tissue_map == tissue_id] = 255
            mask = mask.astype(np.uint8)

            num_labels, output_map, _, _ = cv2.connectedComponentsWithStats(
                mask, 8, cv2.CV_16S)
            for j in range(1, num_labels):
                id = np.unique(map[output_map == j])
                if len(id) > 1:
                    map[output_map == j] = np.min(id)
            # endfor
            return map
        # enddef

        sp_map_merged = merge(tissue_id=1,
                              map=sp_map_merged)   # epithelium merging
        # (stroma + necrosis) merging
        sp_map_merged = merge(tissue_id=2, map=sp_map_merged)
        sp_map_merged = merge(tissue_id=0,
                              map=sp_map_merged)   # background merging

        # ----------------------------------------------------------------------------------------------- Re-arrange labels
        id = np.unique(sp_map_merged)
        for j in range(len(np.unique(sp_map_merged))):
            sp_map_merged[sp_map_merged == id[j]] = j + 1

        return sp_map_merged
    # enddef

    def get_centroids(self, map):
        regions = regionprops(map)
        centroids = np.zeros(shape=(len(regions), 2))

        for i, region in enumerate(regions):
            centroids[i, :] = region['centroid']  # (row, column) = (y, x)
        # endfor
        return centroids
    # enddef

    def extract_sp(self):
        self.load_sp_classifier()

        # for tumor_type in self.tumor_types:
        tumor_type = self.tumor_type

        print('Extracting SP for: ', tumor_type)
        #img_paths = glob.glob(self.base_img_dir + tumor_type + '/*.png')
        img_paths = sorted(glob.glob(self.base_img_dir + '*.png'))

        for i in range(len(img_paths)):
            if i % 20 == 0:
                print(i, '/', len(img_paths))
            self.process_sp(img_paths[i], tumor_type)
        # endfor
        # endfor
    # enddef

    def process_sp(self, img_path, tumor_type):
        basename = os.path.basename(img_path).split('.')[0]
        img_rgb, H, W = self.read_image(img_path)

        # Downsample
        h = int(H / self.magnification)
        w = int(W / self.magnification)
        img = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.blur_kernel_size != 0:
            img = cv2.GaussianBlur(
                img, (self.blur_kernel_size, self.blur_kernel_size), 0)

        # ----------------------------------------------------- UNMERGED SP
        n_segments = min(int(self.base_n_segments * \
                         (h * w / self.base_n_pixels)), self.max_n_segments)
        sp_map_unmerged = slic(
            img,
            sigma=1,
            n_segments=n_segments,
            max_iter=10,
            compactness=20)
        # So that no labeled region is 0 and ignored by regionprops,
        # mark_boundaries
        sp_map_unmerged += 1

        # ----------------------------------------------------- COLOR MERGING
        rag = self.rag_mean_std_color(img, sp_map_unmerged)
        sp_map_merged = graph.merge_hierarchical(
            sp_map_unmerged,
            rag,
            thresh=self.threshold,
            rag_copy=False,
            in_place_merge=True,
            merge_func=self.merge_mean_color,
            weight_func=self.weight_mean_color)
        # So that no labeled region is 0 and ignored by regionprops,
        # mark_boundaries
        sp_map_merged += 1

        # ----------------------------------------------------- RESIZE INSTANCE MAPS
        sp_map_unmerged = cv2.resize(
            sp_map_unmerged, (W, H), interpolation=cv2.INTER_NEAREST)
        sp_map_merged = cv2.resize(
            sp_map_merged, (W, H), interpolation=cv2.INTER_NEAREST)

        # ----------------------------------------------------- CLASSIFIER MERGING
        sp_features_merged = self.extract_sp_classification_features(
            img_rgb=img_rgb, sp_map=sp_map_merged)
        sp_map_merged = self.classification_merge(
            features=sp_features_merged, sp_map=sp_map_merged)

        # ----------------------------------------------------- COMPUTE CENTROIDS
        sp_centroid_unmerged = self.get_centroids(map=sp_map_unmerged)
        sp_centroid_merged = self.get_centroids(map=sp_map_merged)

        # ----------------------------------------------------- SAVE MAPS
        #h5_fout = h5py.File(self.sp_unmerged_detected_path + 'instance_map/' + tumor_type + '/' + basename + '.h5', 'w')
        h5_fout = h5py.File(
            self.sp_unmerged_detected_path +
            'instance_map/' +
            basename +
            '.h5',
            'w')
        h5_fout.create_dataset(
            'detected_instance_map',
            data=sp_map_unmerged,
            dtype='float32')
        h5_fout.close()

        #h5_fout = h5py.File(self.sp_merged_detected_path + 'instance_map/' + tumor_type + '/' + basename + '.h5', 'w')
        h5_fout = h5py.File(
            self.sp_merged_detected_path +
            'instance_map/' +
            basename +
            '.h5',
            'w')
        h5_fout.create_dataset(
            'detected_instance_map',
            data=sp_map_merged,
            dtype='float32')
        h5_fout.close()

        # ----------------------------------------------------- SAVE INSTANCE INFORMATION
        #h5_fout = h5py.File(self.sp_unmerged_detected_path + 'centroids/' + tumor_type + '/' + basename + '.h5', 'w')
        h5_fout = h5py.File(
            self.sp_unmerged_detected_path +
            'centroids/' +
            basename +
            '.h5',
            'w')
        h5_fout.create_dataset(
            'instance_centroid_location',
            data=sp_centroid_unmerged,
            dtype='float32')
        h5_fout.create_dataset(
            'image_dimension',
            data=img_rgb.shape,
            dtype='int32')
        h5_fout.close()

        #h5_fout = h5py.File(self.sp_merged_detected_path + 'centroids/' + tumor_type + '/' + basename + '.h5', 'w')
        h5_fout = h5py.File(
            self.sp_merged_detected_path +
            'centroids/' +
            basename +
            '.h5',
            'w')
        h5_fout.create_dataset(
            'instance_centroid_location',
            data=sp_centroid_merged,
            dtype='float32')
        h5_fout.create_dataset(
            'image_dimension',
            data=img_rgb.shape,
            dtype='int32')
        h5_fout.close()

        #overlaid_plot(img_rgb, sp_map_unmerged, sp_centroid_unmerged)
        #overlaid_plot(img_rgb, sp_map_merged, sp_centroid_merged)
    # enddef
# end
