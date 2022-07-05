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

import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.constants import PLASTICC_WEIGHTS_DICT

# 'SettingWithCopyWarning' in Pandas: https://bit.ly/3mv3fhw
pd.options.mode.chained_assignment = None  # default='warn'

np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.2f}".format})


class ClassWeightedLogLoss(keras.losses.Loss):

    # initialize instance attributes
    def __init__(self, name="class_weighted_log_loss"):
        super().__init__(name=name)

    # compute loss
    def call(self, y_true, y_pred):
        """
        Parameters:
        -----------
        `y_true`: numpy.ndarray

        `y_pred`: numpy.ndarray

        References:
        -----------
        - https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
        def mywloss(y_true, y_pred):
            yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
            loss = -(tf.reduce_mean(tf.reduce_mean(y_true * tf.log(yc), axis=0) / wtable))
            return loss
        Where:
        wtable - is a numpy 1d array with (the number of times class y_true occur in the data set)/(size of data set)
        """
        # array([ 6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95])
        wtable = np.array([2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2]) / y_true.shape[0]

        yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)

        yc = tf.cast(yc, tf.float64)
        y_true = tf.cast(y_true, tf.float64)
        wtable = tf.cast(wtable, tf.float64)

        loss = -(
            tf.reduce_mean(
                tf.math.divide_no_nan(
                    tf.reduce_mean(y_true * tf.math.log(yc), axis=0), wtable
                )
            )
        )

        return loss


class WeightedLogLoss(keras.losses.Loss):

    # initialize instance attributes
    def __init__(self, name="weighted_log_loss"):
        super().__init__(name=name)

    # compute loss
    def call(self, y_true, y_pred):
        """
        Parameters:
        -----------
        y_true: numpy.ndarray
            Actual instance labels

        y_pred: numpy.ndarray
            Predicted values from model.predict() function

        Returns:
        --------
        loss: float64


        References:
        -----------
        - https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
        def mywloss(y_true, y_pred):
            yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
            loss = -(tf.reduce_mean(tf.reduce_mean(y_true * tf.log(yc), axis=0) / wtable))
            return loss
        Where:
        wtable - is a numpy 1d array with (the number of times class y_true occur in the data set)/(size of data set)

        Examples:
        ---------
        >>> print("Running predictions")
        >>> wloss = WeightedLogLoss()
        >>> y_preds = model.predict([X_test, Z_test])
        >>> print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")
        >>> y_preds = model.predict([X_test, Z_test], batch_size=BATCH_SIZE)
        >>> print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")
        """
        wtable = np.sum(y_true, axis=0) / y_true.shape[0]

        yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)

        yc = tf.cast(yc, tf.float64)
        y_true = tf.cast(y_true, tf.float64)
        wtable = tf.cast(wtable, tf.float64)

        loss = -(
            tf.reduce_mean(
                tf.math.divide_no_nan(
                    tf.reduce_mean(y_true * tf.math.log(yc), axis=0), wtable
                )
            )
        )

        return loss


class DistributedWeightedLogLoss(keras.losses.Loss):

    # initialize instance attributes
    def __init__(
        self,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="weighted_log_loss",
    ):
        super().__init__(reduction=reduction, name=name)

    # compute loss
    def call(self, y_true, y_pred):
        """
        Parameters:
        -----------
        y_true: numpy.ndarray
            Actual instance labels

        y_pred: numpy.ndarray
            Predicted values from model.predict() function

        Returns:
        --------
        loss: float64


        References:
        -----------
        - https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
        def mywloss(y_true, y_pred):
            yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
            loss = -(tf.reduce_mean(tf.reduce_mean(y_true * tf.log(yc), axis=0) / wtable))
            return loss
        Where:
        wtable - is a numpy 1d array with (the number of times class y_true occur in the data set)/(size of data set)

        Examples:
        ---------
        >>> print("Running predictions")
        >>> wloss = WeightedLogLoss()
        >>> y_preds = model.predict([X_test, Z_test])
        >>> print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")
        >>> y_preds = model.predict([X_test, Z_test], batch_size=BATCH_SIZE)
        >>> print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")
        """
        wtable = np.sum(y_true, axis=0) / y_true.shape[0]

        yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)

        yc = tf.cast(yc, tf.float64)
        y_true = tf.cast(y_true, tf.float64)
        wtable = tf.cast(wtable, tf.float64)

        loss = -(
            tf.reduce_mean(
                tf.math.divide_no_nan(
                    tf.reduce_mean(y_true * tf.math.log(yc), axis=0), wtable
                )
            )
        )

        # return tf.reduce_sum(loss * (1.0 / self.global_batch_size))

        # return tf.nn.compute_average_loss(
        #     loss, global_batch_size=self.global_batch_size
        # )

        return loss


class WeightedLogLossTF(keras.losses.Loss):
    # initialize instance attributes
    def __init__(self, name="weighted_log_loss"):
        super().__init__(name=name)

    # compute loss
    def call(self, y_true, y_pred):
        """
        Parameters:
        -----------
        `y_true`: numpy.ndarray

        `y_pred`: numpy.ndarray

        References:
        -----------
        - https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
        def mywloss(y_true, y_pred):
            yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
            loss = -(tf.reduce_mean(tf.reduce_mean(y_true * tf.log(yc), axis=0) / wtable))
            return loss
        Where:
        wtable - is a numpy 1d array with (the number of times class y_true occur in the data set)/(size of data set)
        """
        import tensorflow.experimental.numpy as tnp

        wtable = tnp.sum(y_true, axis=0) / y_true.shape[0]
        # wtable = np.sum(y_true.numpy(), axis=0) / y_true.numpy().shape[0]

        yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)

        yc = tf.cast(yc, tf.float64)
        y_true = tf.cast(y_true, tf.float64)
        wtable = tf.cast(wtable, tf.float64)

        loss = -(
            tf.reduce_mean(
                tf.math.divide_no_nan(
                    tf.reduce_mean(y_true * tf.math.log(yc), axis=0), wtable
                )
            )
        )

        return loss


class FlatWeightedLogLoss(keras.losses.Loss):

    # initialize instance attributes
    def __init__(self, name="weighted_log_loss"):
        super().__init__(name=name)

    # compute loss
    def call(self, y_true, y_pred):
        """
        Parameters:
        -----------
        `y_true`: numpy.ndarray

        `y_pred`: numpy.ndarray

        References:
        -----------
        - https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
        def mywloss(y_true, y_pred):
            yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
            loss = -(tf.reduce_mean(tf.reduce_mean(y_true * tf.log(yc), axis=0) / wtable))
            return loss
        Where:
        wtable - is a numpy 1d array with (the number of times class y_true occur in the data set)/(size of data set)
        See Equation 2 K. Boone et al. T is the number of such classes.
        """

        num_classes = y_true.shape[1]
        # Each class has the same weight
        wtable = np.ones(num_classes) / num_classes

        yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)

        yc = tf.cast(yc, tf.float64)
        y_true = tf.cast(y_true, tf.float64)
        wtable = tf.cast(wtable, tf.float64)

        loss = -(
            tf.reduce_mean(
                tf.math.divide_no_nan(
                    tf.reduce_mean(y_true * tf.math.log(yc), axis=0), wtable
                )
            )
        )

        return loss


def custom_log_loss(y_true, y_pred):
    """
    Parameters:
    -----------
    `y_true`: Tensor

    `y_pred`: numpy.ndarray

    References:
    -----------
    - https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
    def mywloss(y_true, y_pred):
        yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
        loss = -(tf.reduce_mean(tf.reduce_mean(y_true * tf.log(yc), axis=0) / wtable))
        return loss
    Where:
    wtable - is a numpy 1d array with (the number of times class y_true occur in the data set)/(size of data set)
    """

    wtable = np.sum(y_true.numpy(), axis=0) / y_true.numpy().shape[0]

    yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)

    yc = tf.cast(yc, tf.float64)
    y_true = tf.cast(y_true, tf.float64)
    wtable = tf.cast(wtable, tf.float64)

    loss = -(
        tf.reduce_mean(
            tf.math.divide_no_nan(
                tf.reduce_mean(y_true * tf.math.log(yc), axis=0), wtable
            )
        )
    )

    return loss


def plasticc_log_loss(y_true, probs):

    """Weighted LogLoss used in the PLAsTiCC kaggle challenge.

    Parameters
    ----------
    y_true: np.array of shape (# samples,)
        Array of the true classes. This may require enc.inverse_transform(y_true) if values are
        currently one-hot encoded
    probs : np.array of shape (# samples, # features)
        Class probabilities for each sample.

    Returns
    -------
    float
        Weighted log loss used for the Kaggle challenge

    References
    ----------
    - https://github.com/jfpuget/Kaggle_PLAsTiCC/blob/master/code/lgb_best.ipynb
    - https://astrorapid.readthedocs.io/en/latest/_modules/astrorapid/classifier_metrics.html
    - https://www.kaggle.com/c/PLAsTiCC-2018/overview/evaluation
    """
    predictions = probs
    labels = np.unique(y_true)

    # sanitize predictions
    epsilon = (
        sys.float_info.epsilon
    )  # this is machine dependent but essentially prevents log(0)
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    predictions = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
    predictions = np.log(predictions)  # logarithm because we want a log loss

    class_logloss, weights = [], []  # initialize the classes logloss and weights
    for i in range(np.shape(predictions)[1]):  # run for each class
        current_label = labels[i]
        result = np.average(predictions[y_true.ravel() == current_label, i])
        # works like a boolean mask to provide results for current class. ravel() required to fix
        # IndexError: result = np.average(predictions[y_true==current_label, i]) # only those
        # events are from that class IndexError: too many indices for array: array is 2-dimensional,
        # but 3 were indexed

        class_logloss.append(result)
        weights.append(PLASTICC_WEIGHTS_DICT[current_label])

    return -1 * np.average(class_logloss, weights=weights)


def custom_tensorflow_plasticc_loss(y_true, y_pred, flip):
    """An attempt to migrate Dan's implementation into tensorflow operations"""

    predictions = y_pred
    labels = np.unique(flip)

    # sanitize predictions
    epsilon = (
        sys.float_info.epsilon
    )  # this is machine dependent but essentially prevents log(0)
    predictions = tf.experimental.numpy.clip(predictions, epsilon, 1.0 - epsilon)
    predictions = (
        predictions / tf.experimental.numpy.sum(predictions, axis=1)[:, tf.newaxis]
    )
    predictions = tf.math.log(predictions)  # logarithm because we want a log loss

    class_logloss, weights = [], []  # initialize the classes logloss and weights
    for i in range(np.shape(predictions)[1]):  # run for each class
        current_label = labels[i]
        result = tf.experimental.numpy.average(predictions[flip == current_label, i])
        # works like a boolean mask to provide results for current class. ravel() required to fix
        # IndexError: result = np.average(predictions[y_true==current_label, i]) # only those
        # events are from that class IndexError: too many indices for array: array is 2-dimensional,
        # but 3 were indexed

        class_logloss.append(result)
        weights.append(PLASTICC_WEIGHTS_DICT[current_label])

    return -1 * tf.experimental.numpy.average(class_logloss, weights=weights)


class CustomLogLoss(keras.losses.Loss):
    def __init__(self, name="plasticc_log_loss"):
        super().__init__(name=name)
        self.dataset = "plasticc"

    def call(self, y_true, y_pred):

        with open(f"{asnwd}/data/{self.dataset}/encoder.joblib", "rb") as eb:
            encoding = joblib.load(eb)

        y_true = encoding.inverse_transform(y_true)

        predictions = y_pred
        labels = np.unique(y_true)

        # sanitize predictions
        epsilon = (
            sys.float_info.epsilon
        )  # this is machine dependent but essentially prevents log(0)
        predictions = tf.experimental.numpy.clip(predictions, epsilon, 1.0 - epsilon)
        predictions = (
            predictions / tf.experimental.numpy.sum(predictions, axis=1)[:, tf.newaxis]
        )
        predictions = tf.math.log(predictions)  # logarithm because we want a log loss

        class_logloss, weights = [], []  # initialize the classes logloss and weights
        for i in range(np.shape(predictions)[1]):  # run for each class
            current_label = labels[i]
            result = tf.experimental.numpy.average(
                predictions[y_true.ravel() == current_label, i]
            )
            # works like a boolean mask to provide results for current class. ravel() required to fix
            # IndexError: result = np.average(predictions[y_true==current_label, i]) # only those
            # events are from that class IndexError: too many indices for array: array is 2-dimensional,
            # but 3 were indexed

            class_logloss.append(result)
            weights.append(PLASTICC_WEIGHTS_DICT[current_label])

        return -1 * tf.experimental.numpy.average(class_logloss, weights=weights)
