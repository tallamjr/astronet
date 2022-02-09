import numpy as np
import pytest
import tensorflow as tf
from astronet.metrics import custom_log_loss, WeightedLogLoss


def test_custom_log_loss():

    x = tf.constant([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y = np.array([[0.9, 0.1, 0.2], [0.9, 0.4, 0.1], [0.2, 0.9, 0.2]])

    logloss = custom_log_loss(x, y).numpy()
    assert pytest.approx(logloss, 0.07024034167855889)

    del logloss

    logloss = custom_log_loss(x, x.numpy())
    assert abs(logloss.numpy()) == 0.0


def test_weighted_log_loss():

    x = tf.constant([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y = np.array([[0.9, 0.1, 0.2], [0.9, 0.4, 0.1], [0.2, 0.9, 0.2]])

    logloss = WeightedLogLoss()
    logloss = logloss.call(y_true=x, y_pred=y).numpy()
    assert pytest.approx(logloss, 0.07024034167855889)

    del logloss

    logloss = WeightedLogLoss()
    logloss = logloss.call(y_true=x, y_pred=x.numpy())
    assert abs(logloss.numpy()) == 0.0
