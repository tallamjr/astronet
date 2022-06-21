import random as python_random

import numpy as np
import pandas as pd
import tensorflow as tf

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
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


def lazy_load_plasticc_wZ(train_or_test: str = "train"):

    if train_or_test == "train":
        X = f"{asnwd}/data/plasticc/test_set/no99/full_test_transformed_df_timesteps_100_X_full_test_no_99.npy"
        Z = f"{asnwd}/data/plasticc/test_set/no99/full_test_transformed_df_timesteps_100_Z_full_test_no_99.npy"
        y = f"{asnwd}/data/plasticc/test_set/no99/full_test_transformed_df_timesteps_100_y_full_test_no_99.npy"
    elif train_or_test == "test_set":
        X = f"{asnwd}/data/plasticc/test_set/no99/full_test_transformed_df_timesteps_100_X_full_test_no_99.npy"
        Z = f"{asnwd}/data/plasticc/test_set/no99/full_test_transformed_df_timesteps_100_Z_full_test_no_99.npy"
        y = f"{asnwd}/data/plasticc/test_set/no99/full_test_transformed_df_timesteps_100_y_full_test_no_99.npy"
    else:
        return -1

    # memmap the file
    X = np.load(X, mmap_mode="r")
    Z = np.load(Z, mmap_mode="r")
    y = np.load(y, mmap_mode="r")

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


def lazy_load_plasticc_noZ(train_or_test: str = "train"):

    if train_or_test == "train":
        X = f"{asnwd}/data/plasticc/test_set/no99/full_test_transformed_df_timesteps_100_X_full_test_no_99.npy"
        y = f"{asnwd}/data/plasticc/test_set/no99/full_test_transformed_df_timesteps_100_y_full_test_no_99.npy"
    elif train_or_test == "test_set":
        X = f"{asnwd}/data/plasticc/test_set/no99/full_test_transformed_df_timesteps_100_X_full_test_no_99.npy"
        y = f"{asnwd}/data/plasticc/test_set/no99/full_test_transformed_df_timesteps_100_y_full_test_no_99.npy"
    else:
        return -1

    # memmap the file
    X = np.load(X, mmap_mode="r")
    y = np.load(y, mmap_mode="r")

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
