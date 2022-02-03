import numpy as np
import tensorflow as tf

from astronet.constants import astronet_working_directory as asnwd
from astronet.utils import load_dataset


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

import random as python_random

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(RANDOM_SEED)

architecture = "t2"
dataset = "plasticc"
X_train, y_train, X_test, y_test, loss, Z_train, Z_test = load_dataset(
    dataset, redshift=True, avocado=None, testset=True
)
print(
    f"""
        X_TRAIN: {X_train.shape}, Y_TRAIN: {y_train.shape},\n
        X_TEST: {X_test.shape}, Y_TEST: {y_test.shape},\n
        Z_TRAIN: {Z_train.shape}, Z_TEST: {Z_test.shape}\n
        """
)

np.save(
    f"{asnwd}/data/plasticc/test_set/infer/X_test.npy", X_test,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/infer/y_test.npy", y_test,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/infer/Z_test.npy", Z_test,
)
