import time
import glob
from utils import *
from skimage.color import rgb2hed
from sklearn.metrics.pairwise import euclidean_distances


class ExtractFeatures:
    def __init__(self, config):
        self.config = config


    def extract_crowdedness(self, centroids, k=10):
        dist = euclidean_distances(centroids, centroids)
        idx = np.argpartition(dist, kth=k+1, axis=-1)
        x = np.take_along_axis(dist, idx, axis=-1)[:, :k+1]
        feat_crowd = np.sum(x, axis=1)/k
        return np.reshape(feat_crowd, newshape=(-1, 1))


    def extract_chromatin(self, img, map):
        img_h = rgb2hed(img)[:, :, 0]        # Hematoxylin channel
        insts_list = list(np.unique(map))
        insts_list.remove(0)  # remove background

        feat_chroma = np.zeros(shape=(len(insts_list), 3))
        for id, inst_id in enumerate(insts_list):
            rows, cols = np.where(map == inst_id)
            pix = img_h[rows, cols]
            feat_chroma[id, 0] = np.mean(pix)
            feat_chroma[id, 1] = np.std(pix)
            feat_chroma[id, 2] = np.median(pix)
        return feat_chroma


    def extract_roundness(self, embeddings):
        area = embeddings[:, 10]
        perimeter = embeddings[:, 13]
        feat_round = (4 * np.pi * area)/ (perimeter ** 2)
        return np.reshape(feat_round, newshape=(-1, 1))


    def extract_ellipticity(self, embeddings):
        major_axis = embeddings[:, 11]
        minor_axis = embeddings[:, 12]
        feat_ellipt = minor_axis/ major_axis
        return np.reshape(feat_ellipt, newshape=(-1, 1))


    def processing(self, img, map, embeddings, centroids):
        start_time = time.time()

        # Get roundness
        feat_round = self.extract_roundness(embeddings)

        # Get ellipticity
        feat_ellipt = self.extract_ellipticity(embeddings)

        # Get crowdedness features
        feat_crowd = self.extract_crowdedness(centroids)

        # Get chromatin features
        feat_chroma = self.extract_chromatin(img, map)

        # Stack
        embeddings = np.hstack((embeddings, feat_round))
        embeddings = np.hstack((embeddings, feat_ellipt))
        embeddings = np.hstack((embeddings, feat_crowd))
        embeddings = np.hstack((embeddings, feat_chroma))
        print('Time= ', round(time.time() - start_time, 3))

        return embeddings


    def extract_feature(self):
        for t in self.config.tumor_types:
            if t != 'dcis':
                continue

            print('Extracting features for: ', t)
            img_paths = sorted(glob.glob(self.config.img_path + t + '/*.png'))
            print(len(img_paths))

            for i in range(len(img_paths)):
                if i % 10 == 0:
                    print(i, '/', len(img_paths))

                basename = os.path.basename(img_paths[i]).split('.')[0]
                instance_map_path = self.config.instance_map_path + t + '/' + basename + '.h5'
                features_path = self.config.features_path + t + '/' + basename + '.h5'
                info_path = self.config.info_path + t + '/' + basename + '.h5'

                # ----------------------------------------------------------------- Read image information
                img = read_image(img_paths[i])
                map = read_instance_map(instance_map_path)
                embeddings = read_features(features_path)
                centroids, labels = read_info(info_path)

                # ----------------------------------------------------------------- Feature extraction
                embeddings = self.processing(img, map, embeddings, centroids)

                # ----------------------------------------------------------------- Feature saving
                h5_fout = h5py.File(features_path, 'w')
                h5_fout.create_dataset('embeddings', data=embeddings, dtype='float32')
                h5_fout.close()

            print('Done\n\n')





