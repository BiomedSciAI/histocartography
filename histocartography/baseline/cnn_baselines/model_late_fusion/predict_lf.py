import numpy as np
import glob
from PIL import Image
from torchvision import transforms
from cnn_model import *

class Predict:
    def __init__(self, config, modelmode):
        self.base_patches_path = config.base_patches_path
        self.model_save_path = config.model_save_path
        self.device = config.device
        self.modelmode = modelmode
        self.batch_size = config.batch_size
        self.patch_scale = config.patch_scale
        self.magnifications = config.magnifications

        self.load_models()
        self.data_transform()


    def load_models(self):
        self.embedding_model = torch.load(self.model_save_path + 'embedding_model_best_' + self.modelmode + '.pt')
        self.classification_model = torch.load(self.model_save_path + 'classification_model_best_' + self.modelmode + '.pt')

        self.embedding_model = self.embedding_model.to(self.device)
        self.classification_model = self.classification_model.to(self.device)

        self.embedding_model.eval()
        self.classification_model.eval()


    def data_transform(self):
        self.transform = transforms.Compose([transforms.Resize(self.patch_scale),
                                                  transforms.CenterCrop(self.patch_scale),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    def get_patches_path(self, troi_id):
        patches_path = [[] for i in range(len(self.magnifications))]
        tumor_type = troi_id.split('_')[1]

        for m in range(len(self.magnifications)):
            paths = sorted(glob.glob(self.base_patches_path + tumor_type + '/' + self.magnifications[m] + '/' + troi_id + '_*.png'))
            patches_path[m].append(paths)

        for m in range(len(self.magnifications)):
            patches_path[m] = sorted([item for sublist in patches_path[m] for item in sublist])

        return patches_path


    def read_image(self, path):
        img_ = Image.open(path)
        img = self.transform(img_)
        img_.close()
        return img


    def predict(self, troi_id):
        pred_embedding = np.array([])
        pred_probabilities = np.array([])

        patches_path = self.get_patches_path(troi_id)
        patches_idx = np.arange(len(patches_path[0]))

        count = 0
        while count < len(patches_idx):
            idx = patches_idx[count: count + self.batch_size]
            patches = [[] for i in range(len(self.magnifications))]

            for i in range(len(idx)):
                for m in range(len(self.magnifications)):
                    patches[m].append(self.read_image(path=patches_path[m][idx[i]]))

            for m in range(len(self.magnifications)):
                patches[m] = torch.stack(patches[m]).to(self.device)

            # ------------------------------------------------------------------- EVAL MODE
            embedding = [[] for i in range(len(self.magnifications))]
            with torch.no_grad():
                for m in range(len(self.magnifications)):
                    embedding_ = self.embedding_model(patches[m])
                    embedding_ = embedding_.squeeze(dim=2)
                    embedding[m] = embedding_.squeeze(dim=2)

                embedding_ = torch.cat(embedding, dim=1)
                probabilities_ = self.classification_model(embedding_)

                if count == 0:
                    pred_embedding = embedding_.cpu().detach().numpy()
                    pred_probabilities = probabilities_.cpu().detach().numpy()
                else:
                    pred_embedding = np.vstack((pred_embedding, embedding_.cpu().detach().numpy()))
                    pred_probabilities = np.vstack((pred_probabilities, probabilities_.cpu().detach().numpy()))

            count += self.batch_size

        if pred_embedding.ndim == 1:
            pred_embedding = np.reshape(pred_embedding, newshape=(1, -1))
            pred_probabilities = np.reshape(pred_probabilities, newshape=(1, -1))

        return pred_embedding, pred_probabilities