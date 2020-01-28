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
    def __init__(self, config):
        self.base_img_dir = config.base_img_dir
        self.base_sp_dir = config.base_sp_dir

        self.base_n_segments = config.base_n_segments
        self.base_n_pixels = config.base_n_pixels
        self.max_n_segments = config.max_n_segments

        self.threshold = 0.01
        self.w_hist = 0.5
        self.w_mean = 0.5
        self.diff_list = []
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

    def extract_basic_superpixels(self, n):
        r = 8            # magnification level
        blur = 3         # blurring kernel size

        sp_save_path = self.base_sp_dir + '1_results_basic_sp/'
        create_directory(sp_save_path + 'sp_img/')
        create_directory(sp_save_path + 'sp_info/')

        img_paths = glob.glob(self.base_img_dir + '*.png')
        img_paths.sort()

        for i in range(len(img_paths)):
            if (i < 20*n) or (i >= 20*(n+1)):
                continue

            print('i: ', i)
            start_time = time.time()
            basename = os.path.basename(img_paths[i]).split('.')[0]

            img_ = Image.open(img_paths[i])
            img_rgb = np.array(img_)
            img_.close()
            (H, W, C) = img_rgb.shape

            h = int(H / r)
            w = int(W / r)
            img = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_NEAREST)

            if blur != 0:
                img = cv2.GaussianBlur(img, (blur, blur), 0)

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
            save_h5(sp_save_path + 'sp_info/' + basename + '.h5', sp_map=sp_map_merged, sp_feat=sp_feat, sp_centroid=sp_centroid)

            # ----------------------------------------------------- PLOT
            overlaid = np.round(mark_boundaries(img_rgb, sp_map_merged, (0, 0, 0)) * 255, 0).astype(np.uint8)
            instance_map = color.label2rgb(sp_map_merged, img_rgb, kind='overlay')
            instance_map = np.round(segmentation.mark_boundaries(instance_map, sp_map_merged, (0, 0, 0)) * 255, 0).astype(np.uint8)
            combo = np.hstack((overlaid, instance_map))
            imageio.imwrite(sp_save_path + 'sp_img/' + basename + '.png', combo)

            print('#', i, ' : ', basename, 'n_segments=', n_segments, ' n_sp=', len(np.unique(sp_map)), ':', len(np.unique(sp_map_merged)),
                  ' time=', round(time.time() - start_time, 2))
        #enddef
    #enddef

    def extract_main_superpixels(self, n, prob_thr=0.8):
        sp_save_path = self.base_sp_dir + '2_results_main_sp/prob_thr_' + str(prob_thr) + '/'
        create_directory(sp_save_path)
        create_directory(sp_save_path + 'sp_img/')
        create_directory(sp_save_path + 'sp_info/')

        basic_sp_info_path = self.base_sp_dir + '1_results_basic_sp/sp_info/'
        sp_classifier_path = self.base_sp_dir + 'sp_classification/sp_classifier/'

        img_paths = glob.glob(self.base_img_dir + '*.png')
        img_paths.sort()

        # ----------------------------------------------------------------------------------------------- Load sp classifier
        model = pickle.load(open(sp_classifier_path + 'svm_model.pkl', 'rb'))
        data = np.load(sp_classifier_path + 'feature_ids.npz')
        indices = data['indices']
        indices = np.sort(indices)
        data = np.load(sp_classifier_path + 'min_max.npz')
        min_max = data['min_max']
        min_max = min_max[:, indices]

        for i in range(len(img_paths)):
            #if (i < 20*n) or (i >= 20*(n+1)):
            #    continue

            print('i: ', i)
            start_time = time.time()
            basename = os.path.basename(img_paths[i]).split('.')[0]

            # ----------------------------------------------------------------------------------------------- Load image data
            img_ = Image.open(img_paths[i])
            img_rgb = np.array(img_)
            img_.close()

            with h5py.File(basic_sp_info_path + basename + '.h5', 'r') as f:
                sp_map = np.array(f.get('sp_map')[:]).astype(int)
                feats = np.array(f.get('sp_features')[:])

            feats = feats[:, indices]
            for i in range(feats.shape[1]):
                minm = min_max[0, i]
                maxm = min_max[1, i]
                if maxm - minm != 0:
                    feats[:, i] = (feats[:, i] - minm) / (maxm - minm)
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

            overlaid = np.round(mark_boundaries(img_rgb, sp_map_new, (0, 0, 0)) * 255, 0).astype(np.uint8)
            instance_map = color.label2rgb(sp_map_new, img_rgb, kind='overlay')
            instance_map = np.round(segmentation.mark_boundaries(instance_map, sp_map_new, (0, 0, 0)) * 255, 0).astype(np.uint8)

            combo = np.hstack((overlaid, instance_map))
            imageio.imwrite(sp_save_path + 'sp_img/' + basename + '.png', combo)

            # ----------------------------------------------------------------------------------------------- FEATURES (at original resolution)
            sp_feat, sp_centroid = extract_main_sp_features(img_rgb=img_rgb, sp_map=sp_map_new)
            save_h5(sp_save_path + 'sp_info/' + basename + '.h5', sp_map=sp_map_new, sp_feat=sp_feat, sp_centroid=sp_centroid)

            print('#', i, ' : ', basename, ' reduction:', len(np.unique(sp_map)), ':', len(np.unique(sp_map_new)), ' time=', round(time.time() - start_time, 2))
        #endfor
    #enddef




