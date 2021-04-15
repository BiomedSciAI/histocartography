import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class HoverNet(nn.Module):

    def __init__(self):
        """
        HoverNet PyTorch re-implementation based:
        `HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images`.
        """

        super(HoverNet, self).__init__()

        # define encoder
        self.encode = Encoder()

        # define decoder(s)
        self.decode_np = Decoder()
        self.decode_hv = Decoder()

        # nuclei pixels (NP)
        self.conv_out_np = Conv2dWithActivation(
            64, 2, 1, activation=None, padding=0, bias=True)
        self.preact_out_np = BNReLU(64)

        # horizontal-vertival (HV)
        self.conv_out_hv = Conv2dWithActivation(
            64, 2, 1, activation=None, padding=0, bias=True)
        self.preact_out_hv = BNReLU(64)

        # upsample
        self.upsample2x = Upsample2x()

    def forward(self, images):
        """
        Forward pass.
        """

        # 1. encode
        d = self.encode(images)

        # 2. crop conv maps output 0 and 1
        d[0] = crop_op(d[0], (92, 92))
        d[1] = crop_op(d[1], (36, 36))

        # 3. decode
        np_feat = self.decode_np(d)
        npx = self.preact_out_np(np_feat[-1])

        hv_feat = self.decode_hv(d)
        hv = self.preact_out_hv(hv_feat[-1])

        # 4.1 nuclei pixel (NP)
        logi_np = self.conv_out_np(npx)

        logi_np = logi_np.permute([0, 2, 3, 1])
        soft_np = F.softmax(logi_np, dim=-1)
        prob_np = torch.unsqueeze(soft_np[:, :, :, 1], dim=-1)

        # 4.2 horizontal vertical (HV)
        logi_hv = self.conv_out_hv(hv)
        logi_hv = logi_hv.permute([0, 2, 3, 1])
        prob_hv = logi_hv
        pred_hv = prob_hv

        # 5. concat output
        predmap_coded = torch.cat([prob_np, pred_hv], dim=-1)

        return predmap_coded


class Encoder(nn.Module):

    def __init__(self):
        """
        Encoder.
        """
        super(Encoder, self).__init__()
        # padding of 3 allows to keep the same dimensions
        self.conv0 = Conv2dWithActivation(
            3, 64, 7, activation='bnrelu', padding=3)
        self.group0 = ResidualBlock(64, [64, 64, 256], [1, 3, 1], 3, strides=1)
        self.group1 = ResidualBlock(
            256, [
                128, 128, 512], [
                1, 3, 1], 4, strides=2)
        self.group2 = ResidualBlock(
            512, [
                256, 256, 1024], [
                1, 3, 1], 6, strides=2)
        self.group3 = ResidualBlock(
            1024, [
                512, 512, 2048], [
                1, 3, 1], 3, strides=2)
        self.conv_bot = Conv2dWithActivation(
            2048, 1024, 1, activation=None)   # @TODO: add 'same' padding

    def forward(self, x):

        x1 = self.conv0(x)
        x2 = self.group0(x1)
        x3 = self.group1(x2)
        x4 = self.group2(x3)
        x5 = self.group3(x4)
        x6 = self.conv_bot(x5)
        return [x2, x3, x4, x6]


class SamepaddingLayer(nn.Module):
    """
    Same padding layer. Equivalent to TF `padding=same` conv.
    """

    def __init__(self, ksize, stride):
        super(SamepaddingLayer, self).__init__()
        self.ksize = ksize
        self.stride = stride

    def forward(self, x):
        if x.shape[2] % self.stride == 0:
            pad = max(self.ksize - self.stride, 0)
        else:
            pad = max(self.ksize - (x.shape[2] % self.stride), 0)

        if pad % 2 == 0:
            pad_val = pad // 2
            padding = (pad_val, pad_val, pad_val, pad_val)
        else:
            pad_val_start = pad // 2
            pad_val_end = pad - pad_val_start
            padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)
        x = F.pad(x, padding, "constant", 0)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, ch_in, ch, ksize, count, split=1, strides=1):
        """
        Residual Block.
        """
        super(ResidualBlock, self).__init__()
        self.count = count
        for i in range(0, count):
            if i != 0:
                setattr(self, 'block' + str(i) + '_preact', BNReLU(ch[2]))
            setattr(self,
                    'block' + str(i) + '_conv1',
                    Conv2dWithActivation(ch_in if i == 0 else ch[-1],
                                         ch[0],
                                         ksize[0],
                                         activation='bnrelu'))
            setattr(
                self,
                'block' +
                str(i) +
                '_conv2_pad',
                SamepaddingLayer(
                    ksize[1],
                    stride=strides if i == 0 else 1))
            setattr(
                self,
                'block' + str(i) + '_conv2',
                Conv2dWithActivation(
                    ch[0],
                    ch[1],
                    ksize[1],
                    activation='bnrelu',
                    stride=strides if i == 0 else 1))
            setattr(
                self,
                'block' +
                str(i) +
                '_conv3',
                Conv2dWithActivation(
                    ch[1],
                    ch[2],
                    ksize[2],
                    activation=None))
            if i == 0:
                setattr(
                    self,
                    'block' +
                    str(i) +
                    '_convshortcut',
                    Conv2dWithActivation(
                        ch_in,
                        ch[2],
                        1,
                        stride=strides,
                        activation=None))
        self.bnlast = BNReLU(ch[2])

    def forward(self, in_feats):

        if hasattr(self, 'block0_convshortcut'):
            shortcut = getattr(self, 'block0_convshortcut')(in_feats)
        else:
            shortcut = in_feats

        for i in range(0, self.count):
            out_feats = in_feats
            if i != 0:
                out_feats = getattr(
                    self, 'block' + str(i) + '_preact')(out_feats)

            out_feats = getattr(self, 'block' + str(i) + '_conv1')(out_feats)
            out_feats = getattr(
                self, 'block' + str(i) + '_conv2_pad')(out_feats)
            out_feats = getattr(self, 'block' + str(i) + '_conv2')(out_feats)
            out_feats = getattr(self, 'block' + str(i) + '_conv3')(out_feats)

            in_feats = out_feats + shortcut
            shortcut = in_feats

        out = self.bnlast(in_feats)
        return out


class Upsample2x(nn.Module):

    def __init__(self):
        """
        Usampling input by 2x.
        """
        super(Upsample2x, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.upsample(x)
        return x


class Decoder(nn.Module):

    def __init__(self):
        """
        Decoder.
        """
        super(Decoder, self).__init__()

        # variables with name starting by u3
        self.u3_rz = Upsample2x()
        self.u3_conva = Conv2dWithActivation(1024, 256, 3, activation=None)
        self.u3_dense = DenseBlock(256, [128, 32], [1, 3], 8, split=4)
        self.u3_convf = Conv2dWithActivation(
            512, 512, 1, activation=None, padding=0)

        # variables with name starting by u2
        self.u2_rz = Upsample2x()
        self.u2_conva = Conv2dWithActivation(512, 128, 3, activation=None)
        self.u2_dense = DenseBlock(128, [128, 32], [1, 3], 4, split=4)
        self.u2_convf = Conv2dWithActivation(
            256, 256, 1, activation=None, padding=0)

        # variables with name starting by u1
        self.u1_rz = Upsample2x()
        self.u1_conva_pad = SamepaddingLayer(3, stride=1)
        self.u1_conva = Conv2dWithActivation(256, 64, 3, activation=None)

    def forward(self, x):

        # processing u3
        u3 = self.u3_rz(x[-1])
        u3_sum = u3 + x[-2]
        u3 = self.u3_conva(u3_sum)
        u3 = self.u3_dense(u3)
        u3 = self.u3_convf(u3)

        # processing u2
        u2 = self.u2_rz(u3)
        u2_sum = u2 + x[-3]
        u2x = self.u2_conva(u2_sum)
        u2 = self.u2_dense(u2x)
        u2 = self.u2_convf(u2)

        u1 = self.u1_rz(u2)
        u1_sum = u1 + x[-4]
        u1 = self.u1_conva_pad(u1_sum)
        u1 = self.u1_conva(u1)

        return [u3, u2x, u1]


class DenseBlock(nn.Module):

    def __init__(self, ch_in, ch, ksize, count, split=1):
        """
        DenseBlock.
        """
        super(DenseBlock, self).__init__()
        self.count = count
        for i in range(0, count):
            setattr(self, 'blk_' + str(i) + 'preact_bna', BNReLU(ch_in))
            setattr(
                self,
                'blk_' +
                str(i) +
                'conv1',
                Conv2dWithActivation(
                    ch_in,
                    ch[0],
                    ksize[0],
                    activation='bnrelu'))
            setattr(
                self,
                'blk_' + str(i) + 'conv2',
                Conv2dWithActivation(
                    ch[0],
                    ch[1],
                    ksize[1],
                    activation=None,
                    split=split))
            ch_in = ch_in + ch[-1]

        self.blk_bna = BNReLU(ch_in)

    def forward(self, l):
        for i in range(0, self.count):
            x = getattr(self, 'blk_' + str(i) + 'preact_bna')(l)
            x = getattr(self, 'blk_' + str(i) + 'conv1')(x)
            x = getattr(self, 'blk_' + str(i) + 'conv2')(x)
            x_shape = list(x.shape)
            l_shape = list(l.shape)
            l = crop_op(l, (l_shape[2] - x_shape[2],
                            l_shape[3] - x_shape[3]))
            l = torch.cat([l, x], dim=1)
        l = self.blk_bna(l)
        return l


class BNReLU(nn.Module):

    def __init__(self, num_features):
        """
        BNReLU.
        """
        super(BNReLU, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2dWithActivation(nn.Module):

    def __init__(
            self,
            num_input,
            num_output,
            filter_size,
            stride=1,
            activation=None,
            padding=0,
            bias=False,
            split=1):
        """
        Conv2dWithActivation.
        """
        super(Conv2dWithActivation, self).__init__()
        self.conv = nn.Conv2d(
            num_input,
            num_output,
            filter_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=split)
        self.activation = activation

        if self.activation is not None:
            if activation == 'bnrelu':
                self.act = BNReLU(num_output)
            else:
                raise ValueError('Not implemented')

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.act(x)
        return x


def crop_op(x, cropping):
    """
    Center crop image.
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    return x
