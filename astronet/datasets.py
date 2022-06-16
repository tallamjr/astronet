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
