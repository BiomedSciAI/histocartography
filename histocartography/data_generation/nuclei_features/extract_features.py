import numpy as np
import glob
import h5py
import cv2
import scipy
from PIL import Image
from torchvision import transforms
from skimage.feature import greycomatrix, greycoprops
from skimage.filters.rank import entropy as Entropy
from skimage.morphology import disk
from utils import *
from get_cnn_model import *


class Extract_HC_Features:
    def __init__(self, config):
        self.base_img_dir = config.base_img_dir
        self.nuclei_detected_path = config.nuclei_detected_path
        self.nuclei_features_path = config.nuclei_features_path
        self.tumor_types = config.tumor_types

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
            nuclei_instance_map = np.array(f['detected_instance_map'])

        return nuclei_instance_map
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

    def processing(self, img, map):
        img = np.full(map.shape + (3,), 200,
                      dtype=np.uint8) if img is None else np.copy(img)

        insts_list = list(np.unique(map))
        insts_list.remove(0)  # remove background

        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        node_feat = []
        entropy = Entropy(img_g, disk(3))

        for idx, inst_id in enumerate(insts_list):
            inst_map = np.array(map == inst_id, np.uint8)
            nuc_feat = []

            # get bounding box for each nuclei
            y1, y2, x1, x2 = self.bounding_box(inst_map)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= map.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= map.shape[0] - 1 else y2
            nuclei_map = inst_map[y1:y2, x1:x2]
            nuclei_img_g = img_g[y1:y2, x1:x2]
            nuclei_entropy = entropy[y1:y2, x1:x2]

            background_px = np.array(nuclei_img_g[nuclei_map == 0])
            foreground_px = np.array(nuclei_img_g[nuclei_map > 0])

            # Morphological features (mean_fg, diff, var, skew)
            mean_bg = background_px.sum() / (np.size(background_px) + 1.0e-8)
            mean_fg = foreground_px.sum() / (np.size(foreground_px) + 1.0e-8)
            diff = abs(mean_fg - mean_bg)
            var = np.var(foreground_px)
            skew = scipy.stats.skew(foreground_px)

            # Textural features (gray level co-occurence matrix)
            glcm = greycomatrix(nuclei_img_g * nuclei_map, [1], [0])
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

            mean_entropy = cv2.mean(nuclei_entropy, mask=nuclei_map)[0]

            _, contours, _ = cv2.findContours(
                nuclei_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]

            num_vertices = len(contour)
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                hull_area += 1
            solidity = float(area) / hull_area
            if num_vertices > 4:
                centre, axes, orientation = cv2.fitEllipse(contour)
                majoraxis_length = max(axes)
                minoraxis_length = min(axes)
            else:
                orientation = 0
                majoraxis_length = 1
                minoraxis_length = 1
            perimeter = cv2.arcLength(contour, True)
            eccentricity = np.sqrt(
                1 - (minoraxis_length / majoraxis_length) ** 2)

            nuc_feat.append(mean_fg)
            nuc_feat.append(diff)
            nuc_feat.append(var)
            nuc_feat.append(skew)
            nuc_feat.append(mean_entropy)
            nuc_feat.append(glcm_dissimilarity)
            nuc_feat.append(glcm_homogeneity)
            nuc_feat.append(glcm_energy)
            nuc_feat.append(glcm_ASM)
            nuc_feat.append(eccentricity)
            nuc_feat.append(area)
            nuc_feat.append(majoraxis_length)
            nuc_feat.append(minoraxis_length)
            nuc_feat.append(perimeter)
            nuc_feat.append(solidity)
            nuc_feat.append(orientation)

            features = np.hstack(nuc_feat)
            node_feat.append(features)
        # endfor

        node_feat = np.vstack(node_feat)
        return node_feat
    # endfor

    def extract_features(self, chunk_id, n_chunks):

        # for tumor_type in self.tumor_types:
        tumor_type = self.tumor_type

        print('Extracting Hand-crafted features for: ', tumor_type)
        img_filepaths = sorted(
            glob.glob(
                self.base_img_dir +
                tumor_type +
                '/*.png'))

        if chunk_id != -1 and n_chunks != -1:
            idx = np.array_split(np.arange(len(img_filepaths)), n_chunks)
            idx = idx[chunk_id]
            img_filepaths = [img_filepaths[x] for x in idx]
        print('# Files=', len(img_filepaths))

        #bracs_s_files = glob.glob('/dataT/pus/histocartography/Data/BRACS_S/Images_norm/' + tumor_type + '/*.png')
        # bracs_s_files.sort()
        # for i in range(len(bracs_s_files)):
        #    bracs_s_files[i] = os.path.basename(bracs_s_files[i]).split('.')[0]
        #print(len(img_filepaths), ':', len(bracs_s_files))

        for i in range(len(img_filepaths)):
            if i % 20 == 0:
                print(i, '/', len(img_filepaths))

            filename = os.path.basename(img_filepaths[i]).split('.')[0]

            # if filename in bracs_s_files:
            #    continue

            nuclei_map_path = self.nuclei_detected_path + \
                'instance_map/' + tumor_type + '/_h5/' + filename + '.h5'

            if os.path.isfile(
                self.nuclei_features_path +
                tumor_type +
                '/' +
                filename +
                    '.h5'):
                continue

            if os.path.isfile(nuclei_map_path):
                # ----------------------------------------------------------------- Read image information
                # Image
                img, h, w = self.read_image(img_filepaths[i])
                # Nuclei instance map
                map = self.read_instance_map(nuclei_map_path)

                # ----------------------------------------------------------------- Feature extraction
                embeddings = self.processing(img, map)

                # ----------------------------------------------------------------- Feature saving
                h5_fout = h5py.File(
                    self.nuclei_features_path +
                    tumor_type +
                    '/' +
                    filename +
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


class Extract_Deep_Features:
    def __init__(self, config, embedding_dim, network):
        self.base_img_dir = config.base_img_dir
        self.nuclei_detected_path = config.nuclei_detected_path
        self.nuclei_features_path = config.nuclei_features_path
        self.patch_size = config.patch_size
        self.patch_size_2 = int(config.patch_size / 2)
        self.tumor_types = config.tumor_types
        self.batch_size = config.batch_size
        self.device = config.device
        self.embedding_dim = embedding_dim
        self.network = network
        self.is_mask = config.is_mask

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
            nuclei_instance_map = np.array(f['detected_instance_map'])
        return nuclei_instance_map
    # enddef

    def read_centroids(self, centroids_path):
        with h5py.File(centroids_path, 'r') as f:
            centroids = np.array(f['instance_centroid_location']).astype(int)
        return centroids
    # enddef

    def extract_features(self):
        # for tumor_type in self.tumor_types:
        tumor_type = self.tumor_type

        print('Extracting Deep features for: ', tumor_type)
        img_filepaths = glob.glob(self.base_img_dir + tumor_type + '/*.png')
        print(self.base_img_dir + '*.png')

        img_filepaths.sort()
        print('#Files = ', len(img_filepaths))

        for i in range(len(img_filepaths)):
            if i % 20 == 0:
                print(i, '/', len(img_filepaths))

            filename = os.path.basename(img_filepaths[i]).split('.')[0]
            nuclei_map_path = self.nuclei_detected_path + \
                'instance_map/' + tumor_type + '/_h5/' + filename + '.h5'
            nuclei_centroids_path = self.nuclei_detected_path + \
                'centroids/' + tumor_type + '/' + filename + '.h5'

            if os.path.isfile(
                self.nuclei_features_path +
                tumor_type +
                '/' +
                filename +
                    '.h5'):
                continue

            if os.path.isfile(nuclei_map_path) and os.path.isfile(
                    nuclei_centroids_path):
                embeddings = np.zeros(shape=(1, self.embedding_dim))

                # ----------------------------------------------------------------- Read image information
                # Image
                img, h, w = self.read_image(img_filepaths[i])
                img = np.pad(
                    img,
                    ((self.patch_size,
                      self.patch_size),
                     (self.patch_size,
                      self.patch_size),
                        (0,
                         0)),
                    mode='constant',
                    constant_values=-1)

                # Nuclei location
                centroids = self.read_centroids(nuclei_centroids_path)

                if self.is_mask:
                    # Nuclei instance map
                    map = self.read_instance_map(nuclei_map_path)
                    map = np.pad(
                        map,
                        ((self.patch_size,
                          self.patch_size),
                         (self.patch_size,
                          self.patch_size)),
                        mode='constant',
                        constant_values=-1)

                # ----------------------------------------------------------------- Feature extraction
                count = 0
                while count < centroids.shape[0]:
                    centroids_ = centroids[count: count + self.batch_size, :]
                    centroids_ += self.patch_size
                    patches = []

                    for j in range(centroids_.shape[0]):
                        x = centroids_[j, 0]
                        y = centroids_[j, 1]
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
                            nuclei_label = count + j + 1  # added 1 because map contains background nuclei_label=0
                            mask = np.array(
                                mask == nuclei_label, np.uint8) * 255
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

                # ----------------------------------------------------------------- Feature saving
                h5_fout = h5py.File(
                    self.nuclei_features_path +
                    tumor_type +
                    '/' +
                    filename +
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


class Extract_CNN_Features:
    def __init__(self, config):
        self.device = config.device
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


class Extract_VAE_Features:
    def __init__(self, config):
        self.device = config.device
        self.model_save_path = config.model_save_path
        self.load_model()

        if config.encoder == 'None':
            self.data_transform = transforms.ToTensor()
        else:
            self.data_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(
            ), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # enddef

    def load_model(self):
        if torch.cuda.is_available():
            self.vae = torch.load(
                self.model_save_path +
                'vae_model_best_loss.pt')
        else:
            self.vae = torch.load(
                self.model_save_path +
                'vae_model_best_loss.pt',
                map_location=torch.device('cpu'))
        self.vae = self.vae.to(self.device)
        self.vae.eval()
    # enddef

    def predict(self, data, visualize=False):
        with torch.no_grad():
            mu, logvar = self.vae._build_encoder(data)
            z = self.vae._reparametrize(mu, logvar)

            if visualize:
                self.visualize_patches(x=data, z=z)

            embeddings = z.cpu().detach().numpy()
        # end
        return embeddings
    # enddef

    def visualize_patches(self, x, z):
        x_reconstructed = self.vae._build_decoder(z)

        for idx in range(x.shape[0]):
            p = (x[idx, :, :, :].cpu().detach().numpy() * 255).astype(np.uint8)
            r = (x_reconstructed[idx, :, :, :].cpu(
            ).detach().numpy() * 255).astype(np.uint8)

            p = np.moveaxis(p, 0, -1)
            r = np.moveaxis(r, 0, -1)

            if idx == 0:
                combo = np.hstack((p, r))
            else:
                combo_ = np.hstack((p, r))
                combo = np.vstack((combo, combo_))
        # endfor
        Image.fromarray(combo).save(self.model_save_path + 'visualize.png')
        print('Visualization complete!!!')
        exit()
    # enddef
# end
