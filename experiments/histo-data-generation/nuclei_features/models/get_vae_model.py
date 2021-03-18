import torch
import torchvision
import numpy as np
#from histocartography.data_generation.nuclei_features.models.denseVAE import *
#from histocartography.data_generation.nuclei_features.models.pretrainedVAE import *
#from histocartography.data_generation.nuclei_features.models.get_cnn_model import *

from denseVAE import *
from pretrainedVAE import *
from get_cnn_model import *


class VAE:
    def __init__(self, config):
        self.encoder = config.encoder
        self.mode = config.mode
        self.embedding_dim = config.embedding_dim
        self.encoder_layers_per_block = config.encoder_layers_per_block
        self.patch_size = config.patch_size
        self.device = config.device

    def get_vae_model(self):
        if self.encoder == 'None':
            self.vae = dense_vae(
                embedding_dim=self.embedding_dim,
                encoder_layers_per_block=self.encoder_layers_per_block,
                patch_size=self.patch_size,
                device=self.device)

        else:
            encoder, num_features = get_encoding_model(
                encoder=self.encoder, mode=self.mode)
            # self.vae =
