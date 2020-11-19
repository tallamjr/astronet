import numpy as np
import tensorflow as tf


def test_numpy_average():

    numpy_API = np.average(range(1, 11), weights=range(10, 0, -1))

    tensorflow_API = tf.experimental.numpy.average(range(1, 11), weights=range(10, 0, -1))

    assert numpy_API == tensorflow_API
