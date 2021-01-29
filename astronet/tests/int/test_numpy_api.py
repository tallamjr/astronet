import numpy as np
from packaging import version
import pytest
import tensorflow as tf

@pytest.mark.skipif(version.parse(tf.__version__) < version.parse("2.4.0"), reason="Requires tensorflow 2.4.x")
def test_numpy_average():

    numpy_API = np.average(range(1, 11), weights=range(10, 0, -1))
    tensorflow_API = tf.experimental.numpy.average(range(1, 11), weights=range(10, 0, -1))

    assert numpy_API == tensorflow_API
