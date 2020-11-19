import numpy as np
import pytest
import tensorflow as tf
from astronet.t2.metrics import custom_log_loss


def test_custom_log_loss():

    x = tf.constant([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    y = np.array([[0.9, 0.1, 0.2], [0.9, 0.4, 0.1], [0.2, 0.9, 0.2]])

    logloss = custom_log_loss(x, y).numpy()
    assert pytest.approx(logloss, 0.07024034167855889)

    logloss = custom_log_loss(x, x.numpy())
    assert abs(logloss.numpy()) == 0.0
