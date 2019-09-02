"""Tumor Slide Segmentation ."""
import logging
import sys
from torch import nn
from pytorch.unet import UNet

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::ML::Tumor Slide Segmentation')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
h1.setFormatter(formatter)
log.addHandler(h1)


class TumorPatchSegmentation(nn.Module):

    def __init__(self, params=None, *args, **kwargs):
        log.info("Creating new Segmenter")
        self.network = UNet()
        super(TumorPatchSegmentation, self).__init__()

    def forward(self, x):

        return self.network(x)
