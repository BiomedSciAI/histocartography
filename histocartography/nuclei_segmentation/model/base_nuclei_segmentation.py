# Base Nuclei Segmentation using TensorPack (CPU)

from tensorpack import *
import tensorflow as tf
from config import Config


class BaseNucleiSegmentation(ModelDesc, Config):
    """
    Base interface class for Nuclei Segmentation(using tensorpack)
    """
    def __init__(self, freeze=False):
        super(BaseNucleiSegmentation, self).__init__()
        # assert tf.test.is_gpu_available()
        self.freeze = freeze
        self.data_format = 'NHWC'  # for CPU

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None] + self.train_input_shape + [3], 'images'),
                InputDesc(tf.float32, [None] + self.train_mask_shape + [None], 'truemap-coded')]

    # for node to receive manual info such as learning rate.
    def add_manual_variable(self, name, init_value, summary=True):
        var = tf.get_variable(name, initializer=init_value, trainable=False)
        if summary:
            tf.summary.scalar(name + '-summary', var)
        return

    def _get_optimizer(self):
        with tf.variable_scope("", reuse=True):
            lr = tf.get_variable('learning_rate')
        opt = self.optimizer(learning_rate=lr)
        return opt

