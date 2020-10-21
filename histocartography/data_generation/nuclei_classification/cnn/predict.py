import math
from matplotlib.patches import Circle
from torchvision import transforms
from utils import *
from cnn_model import *

class Predict:
    def __init__(self, config, args):
        self.config = config
        self.args = args

        self.model_save_path = self.config.base_model_save_path +  'resnet32_bs256_lr0.001_pt_True_ft_True_wl_True/'
        self.device = self.config.device
        self.get_nuclei_colors()

        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
        self.load_checkpoint(self.args.model_mode)

        self.patch_size_2 = int(self.config.patch_size / 2)
        self.pad = self.patch_size_2

    def get_samplenames(self, tumor_type):
        samplenames = []
        with open(self.config.base_test_samples_path + 'test_list_' + tumor_type + '.txt', 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                if line != '':
                    samplenames.append(line)
        return samplenames

    def get_nuclei_colors(self):
        self.nuclei_labels = copy.deepcopy(self.config.nuclei_labels)
        self.nuclei_colors = copy.deepcopy(self.config.nuclei_colors)

        # Remove nuclei type = 'NA'
        idx = self.nuclei_labels.index(-1)
        del self.nuclei_colors[-idx]

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

    def get_predictions(self, img, centroids):
        generator = self.flow(centroids)
        ctr = int(math.ceil(centroids.shape[0] / self.args.batch_size))

        pred_labels = np.array([])

        with torch.no_grad():
            for i in range(ctr):
                centroids_ = generator.__next__()
                patches = self.read_patches(img, centroids_)
                patches = patches.to(self.device)

                embedding = self.embedding_model(patches)
                outputs = self.classification_model(embedding)

                pred_labels_ = torch.argmax(outputs, dim=1)
                pred_labels = np.concatenate((pred_labels, pred_labels_.cpu().detach().numpy()))

        return pred_labels

    def visualize(self, img, basename):
        tumor_type = basename.split('_')[1]

        centroids, labels, img_dim = read_centroids(self.config.base_test_save_path + tumor_type + '/' + basename + '.h5', is_label=True)

        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        ax.imshow(img)

        for i in range(centroids.shape[0]):
            x = centroids[i, 0]
            y = centroids[i, 1]
            circ = Circle((x, y), 5, fill=False, color=self.nuclei_colors[labels[i]], linewidth=2)
            ax.add_patch(circ)

        plt.imshow(img)
        plt.axis('off')

        plt.savefig(self.config.base_test_overlaid_path + tumor_type + '/' + basename + '.png', dpi=600, bbox_inches='tight')
        plt.close()

    def predict(self, is_visualize=False):
        for t in self.config.tumor_types:
            print('Predicting for: ', t)
            create_directory(self.config.base_test_save_path + t)
            create_directory(self.config.base_test_overlaid_path + t)

            samplenames = self.get_samplenames(t)

            for i in range(len(samplenames)):
                if i % 10 == 0:
                    print(i, '/', len(samplenames))

                basename = samplenames[i]
                print(basename)

                img_path = self.config.base_test_img_path + t + '/' + basename + '.png'
                img = read_image(img_path)

                centroid_path = self.config.base_centroid_path + t + '/' + basename + '.h5'
                centroids, img_dim = read_centroids(centroid_path)

                img_pad = np.pad(img, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='constant', constant_values=255)

                pred_labels = self.get_predictions(img_pad, centroids + self.pad)
                # labels: 0=normal, 1=atypical, 2=tumor, 3=stromal, 4=lymphocyte, 5=dead

                save_info(self.config.base_test_save_path + t + '/' + basename + '.h5',
                          keys=['instance_centroid_location', 'instance_centroid_label', 'image_dimension'],
                          values=[centroids, pred_labels, img.shape],
                          dtypes=['float32', 'int32', 'int32'])

                if is_visualize:
                    self.visualize(img, basename)

                exit()

















