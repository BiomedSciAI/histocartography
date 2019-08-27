""" Model Parameters Module """
import torch
import torch.optim as optim
import torch.nn as nn
from .utils import gaussian_mixture, kl_divergence_loss, WeightedBCELoss

OPTIMIZER_FACTORY = {
    'Adadelta': optim.Adadelta,
    'Adagrad': optim.Adagrad,
    'Adam': optim.Adam,
    'Adamax': optim.Adamax,
    'RMSprop': optim.RMSprop,
    'SGD': optim.SGD
}

ACTIVATION_FN_FACTORY = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'tanh': nn.Tanh(),
    'lrelu': nn.LeakyReLU(),
    'elu': nn.ELU()
}
LOSS_FN_FACTORY = {
    'mse': nn.MSELoss(reduction='elementwise_mean'),
    'l1': nn.L1Loss(reduction='elementwise_mean'),
    'mse_sum': nn.MSELoss(reduction='sum'),
    'l1_sum': nn.L1Loss(reduction='sum'),
    'binary_cross_entropy': nn.BCELoss(),
    'bce_logits': nn.BCEWithLogitsLoss(),
    'kld': kl_divergence_loss,
    'weighted_bce': WeightedBCELoss
}

AAE_DISTRIBUTION_FACTORY = {
    'Gaussian': torch.randn,
    'Uniform': torch.rand,
    'Gaussian_Mixture': gaussian_mixture
}
