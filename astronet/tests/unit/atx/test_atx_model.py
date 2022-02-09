import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from astronet.atx.model import ATXModel


def test_num_parameters():
    model = ATXModel(num_classes=14, kernel_size=3, pool_size=3, scaledown_factor=4)
    inputs = tf.keras.Input(shape=[100, 6])
    model(inputs)
    assert np.sum([K.count_params(p) for p in model.trainable_weights]) == 606770
