import tensorflow as tf
from tensorpack import *
from tensorpack.models import BatchNorm, BNReLU, Conv2D, FixedUnPooling
from .base_nuclei_segmentation import BaseNucleiSegmentation
import numpy as np


def upsample2x(name, x):
    """
    :param name: Name of input after upsampling (2x)
    :param x: input that is upsampled (2x)
    :return: Output of Nearest-neighbour upsampling(scale:2x)
    """
    return FixedUnPooling(name, x, 2, unpool_mat=np.ones(
        (2, 2), dtype='float32'), data_format='channels_last')
# enddef


def res_blk(name, l, ch, ksize, count, split=1, strides=1, freeze=False):
    """
    Residual block
    :param name: Name of residual block
    :param l: Input to the residual block
    :param ch: Output channels
    :param ksize: Kernel size
    :param count: Number of residual units in the block
    :param split: To do group convolution. Split=1: Normal convolution
    :param strides: stride
    :param freeze: If true, stops gradient computation
    :return: returns output after each residual block
    """
    ch_in = l.get_shape().as_list()
    # print("In Res Block")
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block' + str(i)):
                x = l if i == 0 else BNReLU('preact', l)
                x = Conv2D('conv1', x, ch[0], ksize[0], activation=BNReLU)
                x = Conv2D('conv2', x, ch[1], ksize[1], split=split,
                           strides=strides if i == 0 else 1, activation=BNReLU)
                x = Conv2D('conv3', x, ch[2], ksize[2], activation=tf.identity)
                if (strides != 1 or ch_in[1] != ch[2]) and i == 0:
                    l = Conv2D('convshortcut', l, ch[2], 1, strides=strides)
                x = tf.stop_gradient(x) if freeze else x
                l = l + x
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l
# enddef


def dense_blk(name, l, ch, ksize, count, split=1, padding='valid'):
    """
    Dense Block for decoder
    :param name: Name of the Dense block
    :param l: Input to the dense block
    :param ch: Output channels
    :param ksize: Kernel size
    :param count: Number of dense units in the dense block
    :param split: Group convolution if >1
    :param padding: Padding style
    :return: Returns output of the dense block
    """
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('blk/' + str(i)):
                # print('In dense block' + str(i))
                x = BNReLU('preact_bna', l)
                x = Conv2D(
                    'conv1',
                    x,
                    ch[0],
                    ksize[0],
                    padding=padding,
                    activation=BNReLU)
                x = Conv2D(
                    'conv2',
                    x,
                    ch[1],
                    ksize[1],
                    padding=padding,
                    split=split)
                ##
                if padding == 'valid':
                    x_shape = x.get_shape().as_list()
                    l_shape = l.get_shape().as_list()
                    l = crop_op(
                        l,
                        (l_shape[1] -
                         x_shape[1],
                            l_shape[2] -
                            x_shape[2]),
                        "NHWC")  # Lau: changed

                l = tf.concat([l, x], axis=3)  # Lau: changed concat
        l = BNReLU('blk_bna', l)
    return l
# enddef


def encoder(i, freeze):
    """
    Pre-activated ResNet50 Encoder
    :param i: Input to the encoder
    :param freeze: If true, gradient computation is stopped.
    :return: returns output where output[0] - Output of first residual block with 3 residual units
    output[1] - Output of second residual block with 4 residual units
    output[2] - Output of third residual block with 6 residual units
    output[3] - Output of fourth residual block with 3 residual units followed by conv 1x1
    """

    d1 = Conv2D(
        'conv0',
        i,
        64,
        7,
        padding='valid',
        strides=1,
        activation=BNReLU)
    d1 = res_blk(
        'group0', d1, [
            64, 64, 256], [
            1, 3, 1], 3, strides=1, freeze=freeze)

    d2 = res_blk(
        'group1', d1, [
            128, 128, 512], [
            1, 3, 1], 4, strides=2, freeze=freeze)
    d2 = tf.stop_gradient(d2) if freeze else d2

    d3 = res_blk(
        'group2', d2, [
            256, 256, 1024], [
            1, 3, 1], 6, strides=2, freeze=freeze)
    d3 = tf.stop_gradient(d3) if freeze else d3

    d4 = res_blk(
        'group3', d3, [
            512, 512, 2048], [
            1, 3, 1], 3, strides=2, freeze=freeze)
    d4 = tf.stop_gradient(d4) if freeze else d4

    d4 = Conv2D('conv_bot', d4, 1024, 1, padding='same')
    #print("Encoder done")
    return [d1, d2, d3, d4]
# enddef


def decoder(name, i):
    """
    :param name: Name of decoder
    :param i: Input to the decoder
    :return: returns output where output[0]- output of first dense block with 8 dense units after nearest neighbour
    upsampling(2x) with 5x5 and 1x1 conv
    output[1] - output of first dense block after upsampling(2x) and 5x5 conv
    output[2] - Output of second dense block after upsampling(2x), adding with encoder output followed by conv 5x5
    """
    pad = 'valid'  # to prevent boundary artifacts
    with tf.variable_scope(name):
        with tf.variable_scope('u3'):
            u3 = upsample2x('rz', i[-1])
            u3_sum = tf.add_n([u3, i[-2]])

            u3 = Conv2D('conva', u3_sum, 256, 5, strides=1, padding=pad)
            u3 = dense_blk(
                'dense', u3, [
                    128, 32], [
                    1, 5], 8, split=4, padding=pad)
            u3 = Conv2D('convf', u3, 512, 1, strides=1)
            ####
        with tf.variable_scope('u2'):
            u2 = upsample2x('rz', u3)
            u2_sum = tf.add_n([u2, i[-3]])

            u2x = Conv2D('conva', u2_sum, 128, 5, strides=1, padding=pad)
            u2 = dense_blk(
                'dense', u2x, [
                    128, 32], [
                    1, 5], 4, split=4, padding=pad)
            u2 = Conv2D('convf', u2, 256, 1, strides=1)
            ####
        with tf.variable_scope('u1'):
            u1 = upsample2x('rz', u2)
            u1_sum = tf.add_n([u1, i[-4]])

            u1 = Conv2D('conva', u1_sum, 64, 5, strides=1, padding='same')

    return [u3, u2x, u1]
# enddef


def crop_op(x, cropping, data_format='channels_first'):
    """
    Center-cropping image
    :param x: image to be cropped
    :param cropping: [H,W] to be cropped out
    :param data_format: if Channels_first, image dimension is [C,H,W].
    If channels_last, image dimension if [H,W,C]
    :return: returns the center cropped image
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == 'channels_first':
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r]
    return x
# enddef


class Model_NP_HV(BaseNucleiSegmentation):
    def _build_graph(self, inputs):
        images, truemap_coded = inputs

        with argscope(Conv2D, activation=tf.identity, use_bias=False,  # K.he initializer
                      W_init=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')), \
                argscope([Conv2D, BatchNorm], data_format=self.data_format):

            # i = tf.transpose(images, [0, 3, 1, 2]) #LAU: NHWC format : no need to transpose
            # print("Loading images")
            i = images
            i = i if not self.input_norm else i / 255.0

            # print("computing encoder")
            d = encoder(i, self.freeze)
            d[0] = crop_op(d[0], (184, 184), "NHWC")
            d[1] = crop_op(d[1], (72, 72), "NHWC")

            # print("decoder for NP")
            np_feat = decoder('np', d)
            npx = BNReLU('preact_out_np', np_feat[-1])

            # print("decoder for HV")

            hv_feat = decoder('hv', d)
            hv = BNReLU('preact_out_hv', hv_feat[-1])

            if self.type_classification:
                # print("decoder for TP")
                tp_feat = decoder('tp', d)
                tp = BNReLU('preact_out_tp', tp_feat[-1])

                # Nuclei Type Pixels (TP)
                logi_class = Conv2D(
                    'conv_out_tp',
                    tp,
                    self.nr_types,
                    1,
                    use_bias=True,
                    activation=tf.identity)
                soft_class = tf.nn.softmax(logi_class, axis=-1)

            # Nuclei Pixels (NP)
            logi_np = Conv2D(
                'conv_out_np',
                npx,
                2,
                1,
                use_bias=True,
                activation=tf.identity)
            soft_np = tf.nn.softmax(logi_np, axis=-1)
            prob_np = tf.identity(soft_np[..., 1], name='predmap-prob-np')
            prob_np = tf.expand_dims(prob_np, axis=-1)

            # Horizontal-Vertival (HV)
            logi_hv = Conv2D(
                'conv_out_hv',
                hv,
                2,
                1,
                use_bias=True,
                activation=tf.identity)
            pred_hv = tf.identity(logi_hv, name='predmap-hv')

            # * channel ordering: type-map, segmentation map
            if self.type_classification:
                predmap_coded = tf.concat(
                    [soft_class, prob_np, pred_hv], axis=-1, name='predmap-coded')
            else:
                predmap_coded = tf.concat(
                    [prob_np, pred_hv], axis=-1, name='predmap-coded')
        ####
    # enddef
# end
