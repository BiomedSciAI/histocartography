"""Unet impemented as torch.nn.Module."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        number_of_filters=[2, 4, 8, 16, 32],
        activation_fn=nn.ReLU(),
        padding=True,
        batch_normalization=False,
        dropout=0.0
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Adapted from https://github.com/jvanvugt/pytorch-unet
        Args:
            input_channels (int): number of input channels.
            output_channels (int): number of output channels.
            number_of_filters (list): channels on each level (matched up down),
                len is depth of the network.
            activation_fn (activation function): an activation function for
                the inner layers. Defaults to nn.ReLU.
            padding (bool): if True, apply padding such that the input shape
                is the same as the output. This may introduce artifacts.
                Defaults to True.
            batch_normalization (bool): Use BatchNorm after layers with an
                activation function. Defaults to False.
            dropout (float): dropout rate. Defaults to 0.0.
        """
        # wf (int): number of filters in the first layer is 2**wf
        #   not used, see number_of_filters
        super(UNet, self).__init__()
        depth = len(number_of_filters)

        prev_channels = input_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            next_channels = number_of_filters[i]
            self.down_path.append(
                UNetConvBlock(
                    prev_channels, next_channels, activation_fn, padding,
                    dropout, batch_normalization
                )
            )
            prev_channels = number_of_filters[i]

        self.up_path = nn.ModuleList()

        for i in reversed(range(depth - 1)):
            next_channels = number_of_filters[i]
            self.up_path.append(
                UNetUpBlock(
                    prev_channels, next_channels, activation_fn, padding,
                    dropout, batch_normalization
                )
            )
            prev_channels = number_of_filters[i]

        self.last = nn.Sequential(
            nn.Conv2d(prev_channels, output_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Apply the forward pass of the model.
        Args:
            example: a torch.Tensor representing the example.
        Returns:
            a torch.Tensor with the mapped examples.
        """
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        output = self.last(x)
        return output


class UNetConvBlock(nn.Module):

    def __init__(
        self, input_channels, output_channels, activation_fn, padding, dropout,
        batch_normalization
    ):
        """
        Initialize a Unet block.
        Args:
            input_channels (int): number of input channels.
            output_channels (int): number of output channels.
            activation_fn (activation function): an activation function for
                the inner layers.
            padding (bool): if True, apply padding such that the input shape
                is the same as the output. This may introduce artifacts.
            batch_normalization (bool): Use BatchNorm after layers with an
                activation function.
            dropout (float): dropout rate.
        """
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=int(padding),
                bias=False
            )
        )
        block.append(activation_fn)

        if batch_normalization:
            block.append(nn.BatchNorm2d(output_channels))

        if dropout > 0.0:
            block.append(nn.Dropout(p=dropout))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        """
        Apply the forward pass of the model.
        Args:
            example: a torch.Tensor representing the example.
        Returns:
            a torch.Tensor with the mapped examples.
        """
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):

    def __init__(
        self, input_channels, output_channels, activation_fn, padding, dropout,
        batch_normalization
    ):
        """
        Initialize a Unet Upconv block.
        Args:
            input_channels (int): number of input channels.
            output_channels (int): number of output channels.
            activation_fn (activation function): an activation function for
                the inner layers.
            padding (bool): if True, apply padding such that the input shape
                is the same as the output. This may introduce artifacts.
            batch_normalization (bool): Use BatchNorm after layers with an
                activation function.
            dropout (float): dropout rate.
        """
        super(UNetUpBlock, self).__init__()
        self.up = nn.Sequential(
            # Evaluate if it's better to upsample using transposed convolution
            nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=int(padding),
                bias=False
            )
        )
        input_channels = 2 * output_channels
        self.conv_block = UNetConvBlock(
            input_channels, output_channels, activation_fn, padding, dropout,
            batch_normalization
        )

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]
                                   ), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        """
        Apply the forward pass of the model.
        Args:
            example: a torch.Tensor representing the example.
        Returns:
            a torch.Tensor with the mapped examples.
        """
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out
