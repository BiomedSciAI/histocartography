import math
import glob
import torch
from torchvision import transforms
from utils import *
from cnn_model import *

class ExtractEmbedding:
    def __init__(self, config, args):
        self.config = config
        self.args = args

        self.model_save_path = self.config.base_model_save_path +  'resnet32_bs256_lr0.001_pt_True_ft_True_wl_True/'
        self.device = self.config.device

        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
        self.load_checkpoint(self.args.model_mode)

        self.patch_size_2 = int(self.config.patch_size / 2)
        self.pad = self.patch_size_2


    def load_checkpoint(self, modelmode):
        embedding_checkpoint = torch.load(self.model_save_path + 'embedding_model_best_' + modelmode + '.th', map_location=self.device)
        classification_checkpoint = torch.load(self.model_save_path + 'classification_model_best_' + modelmode + '.th', map_location=self.device)

        components = ModelComponents(self.config, self.args)
        components.get_embedding_model()
        components.get_classification_model()

        modelE = components.embedding_model
        modelC = components.classification_model

        modelE.load_state_dict(embedding_checkpoint['state_dict'], strict=True)
        modelC.load_state_dict(classification_checkpoint['state_dict'], strict=True)

        self.embedding_model = modelE.to(self.device)
        self.classification_model = modelC.to(self.device)

        self.embedding_model.eval()
        self.classification_model.eval()
        self.embedding_model.zero_grad()
        self.classification_model.zero_grad()


    def flow(self, centroids):
        for i in range(0, centroids.shape[0], self.args.batch_size):
            yield centroids[i : i + self.args.batch_size]


    def read_patches(self, img, centroids):
        patches = []

        for i in range(centroids.shape[0]):
            x = centroids[i, 0]
            y = centroids[i, 1]

            patch = img[y - self.patch_size_2: y + self.patch_size_2, x - self.patch_size_2: x + self.patch_size_2, :]
            patch = self.transform(patch)
            patches.append(patch)

        return torch.stack(patches)


    def get_embedding(self, img, centroids):
        generator = self.flow(centroids)
        ctr = int(math.ceil(centroids.shape[0] / self.args.batch_size))
        embedding = np.array([])

        with torch.no_grad():
            for i in range(ctr):
                centroids_ = generator.__next__()
                patches = self.read_patches(img, centroids_)
                patches = patches.to(self.device)

                if i == 0:
                    embedding = self.embedding_model(patches)
                else:
                    embedding_ = self.embedding_model(patches)
                    embedding = torch.cat([embedding, embedding_], dim=0)

        return embedding.cpu().detach().numpy()


    def extract_embedding(self):
        for t in self.config.tumor_types:
            print('Extracting embeddings for: ', t)

            annotation_centroid_paths = glob.glob(self.config.base_annotation_centroid_path + t + '/*.h5')
            annotation_centroid_paths.sort()
            create_directory(self.config.base_annotation_embedding_path + t)

            for i in range(len(annotation_centroid_paths)):
                if i % 10 == 0:
                    print(i, '/', len(annotation_centroid_paths))

                basename = os.path.basename(annotation_centroid_paths[i]).split('.')[0]
                img = read_image(self.config.base_img_path + t + '/' + basename + '.png')
                img_pad = np.pad(img, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='constant', constant_values=255)
                centroids, labels, img_dim = read_centroids(annotation_centroid_paths[i], is_label=True)

                embedding = self.get_embedding(img_pad, centroids + self.pad)
                print(embedding.shape)

                if centroids.shape[0] != embedding.shape[0]:
                    print(basename, centroids.shape, embedding.shape)

                save_info(self.config.base_annotation_embedding_path + t + '/' + basename + '.h5',
                          keys=['instance_centroid_location', 'instance_centroid_label', 'image_dimension', 'instance_embedding'],
                          values=[centroids, labels, img_dim, embedding],
                          dtypes=['float32', 'int32', 'int32', 'float32'])
















