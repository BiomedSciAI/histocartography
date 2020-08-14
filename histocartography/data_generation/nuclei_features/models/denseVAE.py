import torch
import torch.nn as nn
from torch.autograd import Variable
#from histocartography.data_generation.nuclei_features.models import densenet
from densenet import *


class DenseVAE(nn.Module):

    def __init__(
            self,
            embedding_dim,
            encoder_layers_per_block,
            patch_size,
            device):
        super(DenseVAE, self).__init__()

        self._embedding_dim = embedding_dim
        self._encoder_layers_per_block = encoder_layers_per_block
        self._encoder_nb_blocks = 5
        self._encoder_image_size = int(
            patch_size / 2**(self._encoder_nb_blocks - 1))
        self._device = device

        self._build_layers()
    # enddef

    def _build_layers(self):
        self.encoder = DenseNet(initial_filters=-1,
                                layers=self._encoder_layers_per_block,
                                nb_dense_block=self._encoder_nb_blocks,
                                growth_rate=12)

        self._upsample_filters = self.encoder._block_output_filters
        self._upsample_filters = [
            ele for ele in reversed(
                self._upsample_filters)]

        self._encoder_output_dim = self._upsample_filters[0] * \
            self._encoder_image_size * self._encoder_image_size

        self._mu_fc = nn.Linear(self._encoder_output_dim, self._embedding_dim)
        self._var_fc = nn.Linear(self._encoder_output_dim, self._embedding_dim)

        self._decoder_fc = nn.Linear(
            self._embedding_dim,
            self._encoder_output_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(self._upsample_filters[0], self._upsample_filters[1], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self._upsample_filters[1]),
            nn.ReLU(),

            nn.ConvTranspose2d(self._upsample_filters[1], self._upsample_filters[2], kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(self._upsample_filters[2]),
            nn.ReLU(),

            nn.ConvTranspose2d(self._upsample_filters[2], self._upsample_filters[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self._upsample_filters[3]),
            nn.ReLU(),

            nn.ConvTranspose2d(self._upsample_filters[3], self._upsample_filters[4], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self._upsample_filters[4]),
            nn.ReLU(),

            nn.ConvTranspose2d(self._upsample_filters[4], self._upsample_filters[5], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self._upsample_filters[5]),
            nn.ReLU(),

            nn.ConvTranspose2d(self._upsample_filters[5], 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )
    # enddef

    def _build_encoder(self, x):
        encoded = self.encoder(x)
        return self._mu_fc(encoded), self._var_fc(encoded)
    # enddef

    def _reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.FloatTensor(std.size()).normal_()
            eps = Variable(eps).to(self._device)
            return eps.mul(std).add_(mu)
        else:
            return mu
    # enddef

    def _build_decoder(self, z):
        deconv_input = self._decoder_fc(z).view(-1,
                                                self._upsample_filters[0],
                                                self._encoder_image_size,
                                                self._encoder_image_size)

        return self.decoder(deconv_input)
    # enddef

    def forward(self, x):
        mu, logvar = self._build_encoder(x)
        z = self._reparametrize(mu, logvar)
        decoded = self._build_decoder(z)
        return decoded, mu, logvar
    # enddef
# end


def dense_vae(
        embedding_dim,
        encoder_layers_per_block=1,
        patch_size=72,
        device=None):
    model = DenseVAE(
        embedding_dim=embedding_dim,
        encoder_layers_per_block=encoder_layers_per_block,
        patch_size=patch_size,
        device=device)
    return model
# ennddef


if __name__ == '__main__':
    from torchsummary import summary

    model = dense_vae(
        embedding_dim=16,
        encoder_layers_per_block=1,
        patch_size=72)
    summary(model, (3, 72, 72))
