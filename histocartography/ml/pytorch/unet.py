"""Tumor Slide Classification ."""
import logging
import sys
import torch
from torch import nn
import torch.nn.functional as F
from .hyperparams import ACTIVATION_FN_FACTORY

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::ML::Pytorch::UNET')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
h1.setFormatter(formatter)
log.addHandler(h1)


class UNet(nn.Module):

    def __init__(self, params, *args, **kwargs):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        """
        super(UNet, self).__init__()

        input_channels = params.get('input_channels',3)
        depth = params.get('depth',3)
        num_filters = params.get('num_filters',[32, 64, 128])
        output_channels = params.get('output_channels',1)
        activation_fn = params.get('activation_fn', 'relu')
        dropout = params.get('dropout', 0.0)
        batch_norm = params.get('batch_norm', True)
        self.pos_weight = params.get('pos_weight', 1)
        self.reconstruction_loss = params.get('reconstruction_loss', 'bce')

        log.debug("Parameters : %s", params)

        prev_channels = input_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            next_channels = num_filters[i]
            self.down_path.append(
                UNetConvBlock(prev_channels, next_channels, activation_fn,
                              dropout, batch_norm))
            prev_channels = num_filters[i]

        self.up_path = nn.ModuleList()

        for i in reversed(range(depth - 1)):
            next_channels = num_filters[i]
            self.up_path.append(
                UNetUpBlock(prev_channels, next_channels, activation_fn,
                            dropout, batch_norm))
            prev_channels = num_filters[i]

        self.last = nn.Sequential(
            nn.Conv2d(prev_channels, output_channels, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, activation_fn, dropout, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=False))
        block.append(ACTIVATION_FN_FACTORY[activation_fn])

        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        if dropout > 0.0:
            block.append(nn.Dropout(p=dropout))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, activation_fn, dropout, batch_norm):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Sequential(
            # Evaluate if it's better to upsample using a transposed convolution
            nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=False))
        in_size = 2 * out_size
        self.conv_block = UNetConvBlock(in_size, out_size, activation_fn,
                                        dropout, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y +
                                   target_size[0]), diff_x:(diff_x +
                                                            target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
