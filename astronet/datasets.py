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

import random as python_random

import numpy as np
import pandas as pd
import tensorflow as tf

from astronet.utils import astronet_logger

log = astronet_logger(__file__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(RANDOM_SEED)

# 'SettingWithCopyWarning' in Pandas: https://bit.ly/3mv3fhw
pd.options.mode.chained_assignment = None  # default='warn'


def lazy_load_tfdataset_from_numpy(file: str):

    # memmap the file
    numpy_data_memmap = np.load(file, mmap_mode="r")

    # generator function
    def data_generator():
        return iter(numpy_data_memmap)

    # create tf dataset from generator fn
    dataset = tf.data.Dataset.from_generator(
        generator=data_generator,
        output_signature=tf.type_spec_from_value(
            numpy_data_memmap[0]
        ),  # Infer spec from first element
        # output_types=numpy_data_memmap.dtype,  DEPRECATED
        # output_shapes=numpy_data_memmap.shape[1:],  # shape, no batch DEPRECATED
    )

    return dataset


def lazy_load_plasticc_wZ(X, Z, y):

    # generator function
    def generator():
        for x, z, L in zip(X, Z, y):
            yield ({"input_1": x, "input_2": z}, L)

    # create tf dataset from generator fn
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=(
            {
                "input_1": tf.type_spec_from_value(X[0]),
                "input_2": tf.type_spec_from_value(Z[0]),
            },
            tf.type_spec_from_value(y[0]),
        ),
    )

    return dataset


def lazy_load_plasticc_noZ(X, y):

    # generator function
    def generator():
        for x, L in zip(X, y):
            yield (x, L)

    # create tf dataset from generator fn
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=(
            tf.type_spec_from_value(X[0]),
            tf.type_spec_from_value(y[0]),
        ),
    )

    return dataset
