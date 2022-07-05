# Copyright 2020 - 2022
# Author: Tarek Allam Jr.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import tensorflow as tf

from astronet.metrics import WeightedLogLoss, custom_log_loss


def test_custom_log_loss():

    x = tf.constant([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y = np.array([[0.9, 0.1, 0.2], [0.9, 0.4, 0.1], [0.2, 0.9, 0.2]])

    logloss = custom_log_loss(x, y).numpy()
    assert logloss == pytest.approx(0.07024034167855889)

    del logloss

    logloss = custom_log_loss(x, x.numpy())
    assert abs(logloss.numpy()) == 0.0


def test_weighted_log_loss():

    x = tf.constant([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y = np.array([[0.9, 0.1, 0.2], [0.9, 0.4, 0.1], [0.2, 0.9, 0.2]])

    logloss = WeightedLogLoss()
    logloss = logloss.call(y_true=x, y_pred=y).numpy()
    assert logloss == pytest.approx(0.07024034167855889)

    del logloss

    logloss = WeightedLogLoss()
    logloss = logloss.call(y_true=x, y_pred=x.numpy())
    assert abs(logloss.numpy()) == 0.0
