"""Tumor Slide Segmentation ."""
import logging
import sys
import numpy as np
from torch import nn
from .pytorch.unet import UNet
from ..io.wsi import WSI

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::ML::Tumor Slide Segmentation')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
h1.setFormatter(formatter)
log.addHandler(h1)


class TumorSlideSegmentation:

    def __init__(self, params={}, *args, **kwargs):
        log.info("Creating new Segmenter")
        self.network = UNet(params)
        self.params = params
        try:
            self.network.load_state_dict(params.get('weights', {}))
            log.info("Segmentation initialized with weights from State Dict")
        except RuntimeError:
            log.warning('Segmenter network not initalized')

        super(TumorSlideSegmentation, self).__init__()

    def segment_slide(self, wsi):

        size = tuple(self.params.get('patch_size', [256, 256]))
        stride = tuple(self.params.get('patch_stride', [256, 256]))
        mag = self.params.get('magnification', 2.5)

        patch_generator = wsi.patches(size, stride, mag)

        slide_mask = np.zeros_like(wsi.tissue_mask_at(mag))
        for patch in patch_generator:
            loc_x, loc_y, full_x, full_y, x_mag, y_mag, image, labels = patch
            segmentation = self.network(image)
            slide_mask[x_mag:x_mag + size[0], y_mag:y_mag +
                       size[1]] = segmentation

        return (slide_mask)
