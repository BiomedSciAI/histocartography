"""Tumor Slide Classification ."""
import logging
import sys

import torch
from torch import nn
import torch.nn.functional as F


# setup logging
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::ML::TumorSlideClassifier')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
h1.setFormatter(formatter)
log.addHandler(h1)

class TumorSlideClassifier(nn.Module):
    def __init__(self, params=None, *args, **kwargs):
        log.info("Creating new Classifier")
        super(TumorSlideClassifier,self).__init__()

    def forward(self, x):

        return x
