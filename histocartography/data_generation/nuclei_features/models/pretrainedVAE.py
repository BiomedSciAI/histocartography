import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision

class PretrainedVAE(nn.Module):

    def __init__(self, encoder, encoder_output_dim, embedding_dim, device):
        super(PretrainedVAE, self).__init__()

        self._encoder = encoder
        self._encoder_output_dim = encoder_output_dim
        self._embedding_dim = embedding_dim
        self._device = device

        self._encoder_image_size = 9    # Set as per input patch size = 72x72x3
        self._upsample_filters = [8, 16, 32, 64]
        self._decoder_input_dim = self._upsample_filters[0] * self._encoder_image_size * self._encoder_image_size

        self._build_layers()
    #enddef

    def _build_layers(self):
        self._mu_fc = nn.Linear(self._encoder_output_dim, self._embedding_dim)
        self._var_fc = nn.Linear(self._encoder_output_dim, self._embedding_dim)

        self._decoder_fc = nn.Linear(self._embedding_dim, self._decoder_input_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self._upsample_filters[0], self._upsample_filters[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self._upsample_filters[1]),
            nn.ReLU(),

            nn.ConvTranspose2d(self._upsample_filters[1], self._upsample_filters[2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self._upsample_filters[2]),
            nn.ReLU(),

            nn.ConvTranspose2d(self._upsample_filters[2], self._upsample_filters[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self._upsample_filters[3]),
            nn.ReLU(),

            nn.ConvTranspose2d(self._upsample_filters[3], 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )
    #enddef

    def _build_encoder(self, x):
        encoded = self._encoder(x).squeeze()
        return self._mu_fc(encoded), self._var_fc(encoded)
    #enddef

    def _reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.FloatTensor(std.size()).normal_()
            eps = Variable(eps).to(self._device)
            return eps.mul(std).add_(mu)
        else:
            return mu
    #enddef

    def _build_decoder(self, z):
        deconv_input = self._decoder_fc(z).view(-1, self._upsample_filters[0], self._encoder_image_size, self._encoder_image_size)
        return self.decoder(deconv_input)
    #enddef

    def forward(self, x):
        mu, logvar = self._build_encoder(x)
        z = self._reparametrize(mu, logvar)
        decoded = self._build_decoder(z)
        return decoded, mu, logvar
    #enddef
#end

def pretrained_vae(encoder, encoder_output_dim, embedding_dim, device=None):
    model = PretrainedVAE(encoder=encoder, encoder_output_dim=encoder_output_dim, embedding_dim=embedding_dim, device=device)
    return model
#ennddef

if __name__ == '__main__':
    from torchsummary import summary

    encoder = torchvision.models.vgg16_bn(pretrained=True)
    num_features = list(encoder.classifier.children())[-1].in_features
    classifier = list(encoder.classifier.children())[:1]
    encoder.classifier = nn.Sequential(*classifier)

    model = pretrained_vae(encoder=encoder, encoder_output_dim=num_features, embedding_dim=16)

    summary(model, (3, 72, 72))


