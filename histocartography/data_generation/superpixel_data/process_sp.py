import glob
from skimage.segmentation import slic
from PIL import Image
from skimage.segmentation import mark_boundaries
import imageio
from utils_sp import *
from skimage.future.graph import RAG
from skimage import segmentation, color
import pickle

class Process_SP:
    def __init__(self, config, chunk_id):
        self.base_img_dir = config.base_img_dir
        self.base_sp_dir = config.base_sp_dir
        self.basic_sp_path = config.basic_sp_path
        self.main_sp_path = config.main_sp_path
        self.sp_img_path = config.sp_img_path
        self.sp_classifier_path = config.sp_classifier_path
        self.tumor_types = config.tumor_types

        self.base_n_segments = config.base_n_segments
        self.base_n_pixels = config.base_n_pixels
        self.max_n_segments = config.max_n_segments
        self.prob_thr = config.prob_thr

        self.threshold = 0.01
        self.w_hist = 0.5
        self.w_mean = 0.5
        self.diff_list = []
        self.n_chunks = 25

        self.magnification = 8          # magnification level
        self.blur_kernel_size = 3       # blurring kernel size

        self.chunk_id = chunk_id
        self.load_image_paths()
    #enddef

    def load_image_paths(self):
        img_paths = []
        for tt in self.tumor_types:
            paths = glob.glob(self.base_img_dir + tt + '/*.png')
            img_paths += paths
        #endfor
        img_paths.sort()

        chunks = np.array_split(np.arange(len(img_paths)), self.n_chunks)
        chunk = chunks[self.chunk_id]
        self.img_paths = [img_paths[x] for x in chunk]

        print('#Files=', len(self.img_paths))
    #enddef

    def color_features_per_channel(self, img_ch):
        hist, _ = np.histogram(img_ch, bins=np.arange(0, 257, 64))  # 8 bins
        return hist
    #enddef

    def weight_mean_color(self, graph, src, dst, n):
        diff_mean = np.linalg.norm(graph.nodes[dst]['mean'] - graph.nodes[n]['mean'])

        diff_r = np.linalg.norm(graph.nodes[dst]['r'] - graph.nodes[n]['r'])/2
        diff_g = np.linalg.norm(graph.nodes[dst]['g'] - graph.nodes[n]['g'])/2
        diff_b = np.linalg.norm(graph.nodes[dst]['b'] - graph.nodes[n]['b'])/2
        diff_hist = (diff_r + diff_g + diff_b)/3

        diff = self.w_hist * diff_hist + self.w_mean * diff_mean

        return {'weight': diff}
    #enddef

    def merge_mean_color(self, graph, src, dst):
        graph.nodes[dst]['x'] += graph.nodes[src]['x']
        graph.nodes[dst]['N'] += graph.nodes[src]['N']
        graph.nodes[dst]['mean'] = (graph.nodes[dst]['x'] / graph.nodes[dst]['N'])
        graph.nodes[dst]['mean'] = graph.nodes[dst]['mean'] / np.linalg.norm(graph.nodes[dst]['mean'])

        graph.nodes[dst]['y'] = np.vstack((graph.nodes[dst]['y'], graph.nodes[src]['y']))
        graph.nodes[dst]['r'] = self.color_features_per_channel(graph.nodes[dst]['y'][:, 0])
        graph.nodes[dst]['g'] = self.color_features_per_channel(graph.nodes[dst]['y'][:, 1])
        graph.nodes[dst]['b'] = self.color_features_per_channel(graph.nodes[dst]['y'][:, 2])

        graph.nodes[dst]['r'] = graph.nodes[dst]['r'] / np.linalg.norm(graph.nodes[dst]['r'])
        graph.nodes[dst]['g'] = graph.nodes[dst]['r'] / np.linalg.norm(graph.nodes[dst]['g'])
        graph.nodes[dst]['b'] = graph.nodes[dst]['r'] / np.linalg.norm(graph.nodes[dst]['b'])
    #enddef

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
            graph.nodes[current]['y'] = np.vstack((graph.nodes[current]['y'], image[index]))

        for n in graph:
            graph.nodes[n]['mean'] = (graph.nodes[n]['x'] / graph.nodes[n]['N'])
            graph.nodes[n]['mean'] = graph.nodes[n]['mean'] / np.linalg.norm(graph.nodes[n]['mean'])

            graph.nodes[n]['y'] = np.delete(graph.nodes[n]['y'], 0, axis=0)
            graph.nodes[n]['r'] = self.color_features_per_channel(graph.nodes[n]['y'][:, 0])
            graph.nodes[n]['g'] = self.color_features_per_channel(graph.nodes[n]['y'][:, 1])
            graph.nodes[n]['b'] = self.color_features_per_channel(graph.nodes[n]['y'][:, 2])

            graph.nodes[n]['r'] = graph.nodes[n]['r'] / np.linalg.norm(graph.nodes[n]['r'])
            graph.nodes[n]['g'] = graph.nodes[n]['r'] / np.linalg.norm(graph.nodes[n]['g'])
            graph.nodes[n]['b'] = graph.nodes[n]['r'] / np.linalg.norm(graph.nodes[n]['b'])

        for x, y, d in graph.edges(data=True):
            diff_mean = np.linalg.norm(graph.nodes[x]['mean'] - graph.nodes[y]['mean'])/2

            diff_r = np.linalg.norm(graph.nodes[x]['r'] - graph.nodes[y]['r'])/2
            diff_g = np.linalg.norm(graph.nodes[x]['g'] - graph.nodes[y]['g'])/2
            diff_b = np.linalg.norm(graph.nodes[x]['b'] - graph.nodes[y]['b'])/2
            diff_hist = (diff_r + diff_g + diff_b) / 3

            diff = self.w_hist * diff_hist + self.w_mean * diff_mean

            d['weight'] = diff

        return graph
    #enddef

    def extract_basic_superpixels(self, save_fig=False):
        print('Extract basic superpixels...')

        for i in range(len(self.img_paths)):
            start_time = time.time()
            basename = os.path.basename(self.img_paths[i]).split('.')[0]
            tumorname = basename.split('_')[1]

            img_ = Image.open(self.img_paths[i])
            img_rgb = np.array(img_)
            img_.close()
            (H, W, C) = img_rgb.shape

            h = int(H / self.magnification)
            w = int(W / self.magnification)
            img = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_NEAREST)

            if self.blur_kernel_size != 0:
                img = cv2.GaussianBlur(img, (self.blur_kernel_size, self.blur_kernel_size), 0)

            # ----------------------------------------------------- SLIC
            n_segments = min(int(self.base_n_segments * (h * w / self.base_n_pixels)), self.max_n_segments)
            sp_map = slic(img, sigma=1, n_segments=n_segments, max_iter=10, compactness=20)
            sp_map += 1             # So that no labeled region is 0 and ignored by regionprops, mark_boundaries

            # ----------------------------------------------------- MERGING
            rag = self.rag_mean_std_color(img, sp_map)
            sp_map_merged = graph.merge_hierarchical(sp_map, rag, thresh=self.threshold, rag_copy=False,
                                                     in_place_merge=True,
                                                     merge_func=self.merge_mean_color,
                                                     weight_func=self.weight_mean_color)

            sp_map_merged += 1              # So that no labeled region is 0 and ignored by regionprops, mark_boundaries

            # ----------------------------------------------------- SAVE Map and Features
            sp_map_merged = cv2.resize(sp_map_merged, (W, H), interpolation=cv2.INTER_NEAREST)
            sp_feat, sp_centroid = extract_basic_sp_features(img_rgb=img_rgb, sp_map=sp_map_merged)
            save_h5(self.basic_sp_path + tumorname + '/' + basename + '.h5', sp_map=sp_map_merged, sp_feat=sp_feat, sp_centroid=sp_centroid)

            # ----------------------------------------------------- PLOT
            if save_fig:
                #overlaid = np.round(mark_boundaries(img_rgb, sp_map_merged, (0, 0, 0)) * 255, 0).astype(np.uint8)
                instance_map = color.label2rgb(sp_map_merged, img_rgb, kind='overlay')
                instance_map = np.round(segmentation.mark_boundaries(instance_map, sp_map_merged, (0, 0, 0)) * 255, 0).astype(np.uint8)
                #combo = np.hstack((overlaid, instance_map))
                imageio.imwrite(self.sp_img_path + 'basic_sp/' + tumorname + '/' + basename + '.png', instance_map)

            print('#', i, ' : ', basename, 'n_segments=', n_segments, ' n_sp=', len(np.unique(sp_map)), ':', len(np.unique(sp_map_merged)), ' time=', round(time.time() - start_time, 2))
        #enddef
    #enddef


    def extract_main_superpixels(self, save_fig=False):
        print('\n\nExtract main superpixels...')

        # ----------------------------------------------------------------------------------------------- Load sp classifier
        sp_classifier_path = self.sp_classifier_path + 'sp_classifier/'

        model = pickle.load(open(sp_classifier_path + 'svm_model.pkl', 'rb'))
        data = np.load(sp_classifier_path + 'feature_ids.npz')
        indices = data['indices']
        indices = np.sort(indices)
        data = np.load(sp_classifier_path + 'min_max.npz')
        min_max = data['min_max']
        min_max = min_max[:, indices]

        for i in range(len(self.img_paths)):
            start_time = time.time()
            basename = os.path.basename(self.img_paths[i]).split('.')[0]
            tumorname = basename.split('_')[1]

            # ----------------------------------------------------------------------------------------------- Load image data
            img_ = Image.open(self.img_paths[i])
            img_rgb = np.array(img_)
            img_.close()

            sp_map, feats, centroids = load_h5(self.basic_sp_path + tumorname + '/' + basename + '.h5')

            feats = feats[:, indices]
            for j in range(feats.shape[1]):
                minm = min_max[0, j]
                maxm = min_max[1, j]
                if maxm - minm != 0:
                    feats[:, j] = (feats[:, j] - minm) / (maxm - minm)
            #endfor

            # ----------------------------------------------------------------------------------------------- Predict SVM output
            pred = model.predict_proba(feats)
            pred_prob = np.max(pred, axis=1)
            pred_label = np.argmax(pred, axis=1)
            pred_label[pred_prob < self.prob_thr] = -1

            # ----------------------------------------------------------------------------------------------- Generate tissue map
            tissue_map = np.ones_like(sp_map) * -1
            regions = regionprops(sp_map)
            for j, region in enumerate(regions):
                if pred_label[j] != -1:
                    tissue_map[sp_map == region['label']] = pred_label[j]
            #endfor

            # ----------------------------------------------------------------------------------------------- Merge super-pixels
            sp_map_new = copy.deepcopy(sp_map)

            def merge(tissue_id, map):
                mask = np.zeros_like(map)
                mask[tissue_map == tissue_id] = 255
                mask = mask.astype(np.uint8)

                num_labels, output_map, _, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_16S)
                for j in range(1, num_labels):
                    id = np.unique(map[output_map == j])
                    if len(id) > 1:
                        map[output_map == j] = np.min(id)
                #endfor
                return map
            #enddef

            sp_map_new = merge(tissue_id=1, map=sp_map_new)
            sp_map_new = merge(tissue_id=2, map=sp_map_new)
            sp_map_new = merge(tissue_id=0, map=sp_map_new)

            # ----------------------------------------------------- Re-arranging labels in sp_map_new
            id = np.unique(sp_map_new)
            for j in range(len(np.unique(sp_map_new))):
                sp_map_new[sp_map_new == id[j]] = j + 1

            if save_fig:
                #overlaid = np.round(mark_boundaries(img_rgb, sp_map_new, (0, 0, 0)) * 255, 0).astype(np.uint8)
                instance_map = color.label2rgb(sp_map_new, img_rgb, kind='overlay')
                instance_map = np.round(segmentation.mark_boundaries(instance_map, sp_map_new, (0, 0, 0)) * 255, 0).astype(np.uint8)
                #combo = np.hstack((overlaid, instance_map))
                imageio.imwrite(self.sp_img_path + 'main_sp/prob_thr_' + str(self.prob_thr) + '/' + tumorname + '/' + basename + '.png', instance_map)

            # ----------------------------------------------------------------------------------------------- FEATURES
            sp_feat, sp_centroid = extract_main_sp_features(img_rgb=img_rgb, sp_map=sp_map_new)
            save_h5(self.main_sp_path + tumorname + '/' + basename + '.h5', sp_map=sp_map_new, sp_feat=sp_feat, sp_centroid=sp_centroid)

            print('#', i, ' : ', basename, ' reduction:', len(np.unique(sp_map)), ':', len(np.unique(sp_map_new)), ' time=', round(time.time() - start_time, 2))
        #endfor
    #enddef




