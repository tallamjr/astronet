import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.constants import (
    LSST_FILTER_MAP,
    LSST_PB_WAVELENGTHS,
    PLASTICC_CLASS_MAPPING,
    SYSTEM,
)
from astronet.metrics import WeightedLogLoss
from astronet.preprocess import (
    __filter_dataframe_only_supernova,
    __transient_trim,
    generate_gp_all_objects,
    one_hot_encode,
    remap_filters,
    robust_scale,
)

log = astronet_logger(__file__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(RANDOM_SEED)

# 'SettingWithCopyWarning' in Pandas: https://bit.ly/3mv3fhw
pd.options.mode.chained_assignment = None  # default='warn'


def tfdataset_from_numpy(dataset: str = "plasticc"):
    log.critical(f"{inspect.stack()[0].function} -- Not Fully Implemented Yet")

    # memmap the file
    numpy_data_memmap = np.load(file, mmap_mode="r")

    # generator function
    def data_generator():
        return iter(numpy_data_memmap)

    # create tf dataset from generator fn
    dataset = tf.data.Dataset.from_generator(
        generator=data_generator,
        output_types=np.float64,
        output_shapes=example_shape,
    )

    return dataset
