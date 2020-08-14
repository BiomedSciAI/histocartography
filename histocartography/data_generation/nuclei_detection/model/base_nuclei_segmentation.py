# Base Nuclei Segmentation using TensorPack (CPU)

from tensorpack import *
import tensorflow as tf

class BaseNucleiSegmentation(ModelDesc):
    """
    Base interface class for Nuclei Segmentation(using tensorpack)
    """
    def __init__(self, config, freeze=False):
        super(BaseNucleiSegmentation, self).__init__()
        # assert tf.test.is_gpu_available()
        self.freeze = freeze
        self.data_format = 'NHWC'  # for CPU
        self.optimizer = config.optimizer
        self.train_input_shape = config.train_input_shape
        self.train_mask_shape = config.train_mask_shape
        self.input_norm = config.input_norm
        self.type_classification = config.type_classification
        self.nr_types = config.nr_types
    #enddef

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

