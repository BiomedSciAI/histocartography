import glob
import h5py
from PIL import Image
import cv2
import torch
from torchvision import transforms
from get_cnn_model import get_encoding_model

#from histocartography.data_generation.tissue_features.models.get_cnn_model import *
from utils import *


class Extract_Deep_Features:
    def __init__(self, config, embedding_dim, network):
        self.base_img_dir = config.base_img_dir
        self.sp_unmerged_detected_path = config.sp_unmerged_detected_path
        self.sp_merged_detected_path = config.sp_merged_detected_path
        self.sp_unmerged_features_path = config.sp_unmerged_features_path
        self.sp_merged_features_path = config.sp_merged_features_path

        self.tumor_types = config.tumor_types
        self.batch_size = config.batch_size
        self.embedding_dim = embedding_dim
        self.network = network
        self.patch_size = config.patch_size
        self.patch_size_2 = int(config.patch_size / 2)
        self.is_mask = config.is_mask

        # set device
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')

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
            sp_instance_map = np.array(f['detected_instance_map'])
        return sp_instance_map
    # enddef

    def read_centroids(self, centroids_path):
        with h5py.File(centroids_path, 'r') as f:
            centroids = np.array(f['instance_centroid_location']).astype(int)
        return centroids
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

    def extract_features(self):
        # for tumor_type in self.tumor_types:
        tumor_type = self.tumor_type

        print('Extracting Unmerged Deep features for: ', tumor_type)
        img_filepaths = sorted(
            glob.glob(
                self.base_img_dir +
                tumor_type +
                '/*.png'))

        for i in range(len(img_filepaths)):
            if i % 20 == 0:
                print(i, '/', len(img_filepaths))

            basename = os.path.basename(img_filepaths[i]).split('.')[0]
            sp_unmerged_map_path = self.sp_unmerged_detected_path + \
                'instance_map/' + tumor_type + '/' + basename + '.h5'
            sp_unmerged_centroid_path = self.sp_unmerged_detected_path + \
                'centroids/' + tumor_type + '/' + basename + '.h5'
            sp_merged_map_path = self.sp_merged_detected_path + \
                'instance_map/' + tumor_type + '/' + basename + '.h5'

            if os.path.isfile(
                self.sp_merged_features_path +
                tumor_type +
                '/' +
                basename +
                    '.h5'):
                continue

            if os.path.isfile(sp_unmerged_map_path) and os.path.isfile(
                    sp_merged_map_path):
                # ----------------------------------------------------------------- Read image information
                # Image
                img, h, w = self.read_image(img_filepaths[i])
                img_pad = np.pad(
                    img,
                    ((self.patch_size,
                      self.patch_size),
                     (self.patch_size,
                      self.patch_size),
                        (0,
                         0)),
                    mode='constant',
                    constant_values=-1)

                # Unmerged SP location
                unmerged_centroids = self.read_centroids(
                    sp_unmerged_centroid_path)

                # SP instance map
                unmerged_map = self.read_instance_map(sp_unmerged_map_path)
                merged_map = self.read_instance_map(sp_merged_map_path)
                unmerged_map_pad = np.pad(
                    unmerged_map,
                    ((self.patch_size,
                      self.patch_size),
                     (self.patch_size,
                      self.patch_size)),
                    mode='constant',
                    constant_values=-1)

                # ----------------------------------------------------------------- Extract unmerged features
                unmerged_features = self.extract_unmerged_features(
                    img_pad, unmerged_map_pad, unmerged_centroids)
                h5_fout = h5py.File(
                    self.sp_unmerged_features_path +
                    tumor_type +
                    '/' +
                    basename +
                    '.h5',
                    'w')
                h5_fout.create_dataset(
                    'embeddings',
                    data=unmerged_features,
                    dtype='float32')
                h5_fout.close()

                # ----------------------------------------------------------------- Extract merged features
                merged_features = self.extract_merged_features(
                    unmerged_map, merged_map, unmerged_features)
                h5_fout = h5py.File(
                    self.sp_merged_features_path +
                    tumor_type +
                    '/' +
                    basename +
                    '.h5',
                    'w')
                h5_fout.create_dataset(
                    'embeddings', data=merged_features, dtype='float32')
                h5_fout.close()
            # endif
        # endfor

        # endfor
    # enddef

    def extract_unmerged_features(self, img, map, centroids):
        embeddings = np.zeros(shape=(1, self.embedding_dim))

        # ----------------------------------------------------------------- Feature extraction
        count = 0
        while count < centroids.shape[0]:
            centroids_ = centroids[count: count + self.batch_size, :]
            centroids_ += self.patch_size
            patches = []

            for j in range(centroids_.shape[0]):
                x = centroids_[j, 1]
                y = centroids_[j, 0]
                patch = img[y -
                            self.patch_size_2: y +
                            self.patch_size_2, x -
                            self.patch_size_2: x +
                            self.patch_size_2, :]

                if self.is_mask:
                    mask = map[y -
                               self.patch_size_2: y +
                               self.patch_size_2, x -
                               self.patch_size_2: x +
                               self.patch_size_2]
                    sp_label = count + j + 1  # added 1 because sp map begins from 1
                    mask = np.array(mask == sp_label, np.uint8) * 255
                    patch = cv2.bitwise_and(patch, patch, mask=mask)
                # endif

                patch = Image.fromarray(patch)
                patch = self.network.data_transform(patch)
                patches.append(patch)
            # endfor
            patches = torch.stack(patches).to(self.device)

            # ------------------------------------------------------------------- EVAL MODE
            embeddings_ = self.network.predict(patches)
            embeddings = np.vstack((embeddings, embeddings_))
            count += self.batch_size
        # end

        embeddings = np.delete(embeddings, 0, axis=0)
        return embeddings
    # enddef

    def extract_merged_features(
            self,
            unmerged_map,
            merged_map,
            unmerged_features):
        inst_ids = list(np.unique(merged_map))
        embeddings = np.zeros(shape=(1, self.embedding_dim))

        for id in inst_ids:
            mask = np.array(merged_map == id, np.uint8)
            mask = mask * unmerged_map

            idx = np.unique(mask)[1:].astype(int)    # remove background
            idx = idx - 1       # because instance map ids start from 1

            feats = unmerged_features[idx, :]
            feats = np.mean(feats, axis=0)
            embeddings = np.vstack((embeddings, feats))
        # endfor
        embeddings = np.delete(embeddings, 0, axis=0)
        return embeddings
    # enddef
# end


class Extract_CNN_Features:
    def __init__(self, config):

        # set device
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')

        self.encoder = config.encoder
        self.mode = config.mode
        self.load_model()

        self.data_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(
        ), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # enddef

    def load_model(self):
        self.cnn, self.num_features = get_encoding_model(
            encoder=self.encoder, mode=self.mode)
        self.cnn = self.cnn.to(self.device)
        self.cnn.eval()
    # enddef

    def predict(self, data):
        with torch.no_grad():
            embeddings = self.cnn(data).squeeze()
            embeddings = embeddings.cpu().detach().numpy()
        # end
        return embeddings
    # enddef
# end
