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
from packaging import version


@pytest.mark.skipif(
    version.parse(tf.__version__) < version.parse("2.4.0"),
    reason="Requires tensorflow 2.4.x",
)
def test_numpy_average():

    numpy_API = np.average(range(1, 11), weights=range(10, 0, -1))
    tensorflow_API = tf.experimental.numpy.average(
        range(1, 11), weights=range(10, 0, -1)
    )

    assert numpy_API == tensorflow_API
