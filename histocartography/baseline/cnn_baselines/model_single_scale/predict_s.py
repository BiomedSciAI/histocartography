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

        if len(self.magnifications) == 1:
            self.magnification = self.magnifications[0]

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


    def predict(self, troi_id):
        pred_embedding = np.array([])
        pred_probabilities = np.array([])

        tumor_type = troi_id.split('_')[1]
        patches_path = sorted(glob.glob(self.base_patches_path + tumor_type + '/' + self.magnification + '/' + troi_id + '_*.png'))
        patches_idx = np.arange(len(patches_path))

        count = 0
        while count < len(patches_idx):
            idx = patches_idx[count: count + self.batch_size]
            paths = [patches_path[x] for x in idx]
            patches = []

            for path in paths:
                patch = Image.open(path)
                patch = self.transform(patch)
                patches.append(patch)

            patches = torch.stack(patches).to(self.device)

            # ------------------------------------------------------------------- EVAL MODE
            with torch.no_grad():
                embedding_ = self.embedding_model(patches)
                embedding_ = embedding_.squeeze(dim=2)
                embedding_ = embedding_.squeeze(dim=2)
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


