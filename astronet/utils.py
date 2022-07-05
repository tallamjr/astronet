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

import logging
import os
import pickle
from pathlib import Path
from typing import List

import colorama
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

# 'SettingWithCopyWarning' in Pandas: https://bit.ly/3mv3fhw
pd.options.mode.chained_assignment = None  # default='warn'


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # green = colorama.Fore.GREEN
    white = colorama.Fore.WHITE

    # Format of the log message to be printed
    FORMAT = "[%(asctime)s] "
    FORMAT += "{%(filename)s:%(lineno)d} "
    FORMAT += "%(levelname)s "
    FORMAT += "- %(message)s"

    FORMATS = {
        logging.DEBUG: grey + FORMAT + reset,
        logging.INFO: white + FORMAT + reset,
        logging.WARNING: yellow + FORMAT + reset,
        logging.ERROR: red + FORMAT + reset,
        logging.CRITICAL: bold_red + FORMAT + reset,
    }

    def format(self, record):
        # Date format
        DATEFORMAT = "%y-%m-%d %H:%M:%S"

        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=DATEFORMAT)
        return formatter.format(record)


def astronet_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Initialise python logger.

    Parameters
    ----------
    name : str
        Should be __name__ of file or module.
    log_level : str
        levels: DEBUG, INFO, WARNING, ERROR, CRITICAL, OFF

    Returns
    ----------
    logger : logging.Logger
        Python Logger

    Examples
    ----------
    >>> logger.debug('debug message')
    >>> logger.info('info message')
    >>> logger.warning('warn message')
    >>> logger.error('error message')
    >>> logger.critical('critical message')
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Add formatter to ch
    ch.setFormatter(CustomFormatter())
    # Add ch to logger
    logger.addHandler(ch)
    # Do not pass logs to ancestor logger as well, i.e. print once:
    # https://docs.python.org/3/library/logging.html#logging.Logger
    logger.propagate = False

    return logger


def tfrecords():

    # x = train_input[0:5000, :, :]
    # x = tf.convert_to_tensor(x)
    # x = tf.io.serialize_tensor(x)
    # x = x.numpy()
    # x = tf.train.BytesList(value=[x])
    # x = tf.train.Feature(bytes_list=x)

    # y = y_train[0:5000, :]
    # y = tf.convert_to_tensor(y)
    # y = tf.io.serialize_tensor(y)
    # y = y.numpy()
    # y = tf.train.BytesList(value=[y])
    # y = tf.train.Feature(bytes_list=y)

    # xy = tf.train.Features(feature={"x": x, "y": y})
    # xy = tf.train.Example(features=xy)
    # xy = xy.SerializeToString()

    # import pdb

    # pdb.set_trace()
    # pth = "xy.tfrecords"

    # with tf.io.TFRecordWriter(
    #     pth, options=tf.io.TFRecordOptions(compression_type="ZLIB")
    # ) as writer:
    #     writer.write(xy)

    # ds = tf.data.TFRecordDataset(pth, compression_type="ZLIB")

    # def _parse_example(xy):
    #     ex = tf.io.parse_single_example(
    #         xy,
    #         {
    #             "x": tf.io.FixedLenFeature([], tf.string),
    #             "y": tf.io.FixedLenFeature([], tf.string),
    #         },
    #     )
    #     # x = tf.io.parse_tensor(ex["x"], tf.double)
    #     # y = tf.io.parse_tensor(ex["y"], tf.double)
    #     return ex["x"], ex["y"]

    # ds = ds.map(_parse_example)

    # for xy in ds.take(1):
    #     print(xy)

    # train_ds = (
    #     ds.shuffle(1000)
    #     .batch(BATCH_SIZE, drop_remainder=True)
    #     .prefetch(tf.data.AUTOTUNE)
    # )
    # test_ds = ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    # def decode_fn(record_bytes):
    #     return tf.io.parse_single_example(
    #         # Data
    #         record_bytes,
    #         # Schema
    #         {
    #             "x": tf.io.FixedLenFeature([], dtype=tf.float32),
    #             "y": tf.io.FixedLenFeature([], dtype=tf.float32),
    #         },
    #     )

    # for batch in tf.data.TFRecordDataset(["example.tfrecords"]).map(decode_fn):
    #     print("x = {x:.4f},  y = {y:.4f}".format(**batch))
    pass


def find_optimal_batch_size(training_set_length: int) -> int:
    """Determine optimal batch size to use. Ideally leave a large remainder such that the GPU is
    full for most of the time.
    """

    if training_set_length < 10000:
        batch_size_list = [16, 32, 64]
    else:
        # batch_size_list = [96, 128, 256]
        batch_size_list = [2048, 4096]
    ratios = []
    for batch_size in batch_size_list:

        remainder = training_set_length % batch_size

        if remainder == 0:
            batch_size = remainder
        else:
            ratios.append(batch_size / remainder)

    index, ratio = min(enumerate(ratios), key=lambda x: abs(x[1] - 1))

    return batch_size_list[index]


def train_val_test_split(df, cols):
    """Split dataset into train, validation and test set."""

    features = df[cols]
    # column_indices = {name: i for i, name in enumerate(features.columns)}

    n = len(df)
    df_train = df[0 : int(n * 0.8)].copy()
    df_val = df[int(n * 0.8) : int(n * 0.95)].copy()
    df_test = df[int(n * 0.95) :].copy()

    num_features = features.shape[1]

    return df_train, df_val, df_test, num_features


def create_dataset(
    X: pd.DataFrame, y: pd.Series, time_steps: int = 1, step: int = 1
) -> (np.ndarray, np.ndarray):
    """Create dataset from GP interpolated data and splitting according to the timesteps used when
    generating the GP dataframe. This allows for the correct label to be assigned to the
    corresponding X values

    Parameters
    ----------
    X: pd.DataFrame
        Subset of full dataframe containing only the passband columns
    y: pd.Series
        A pd.Series containing the object labels
    TIME_STEPS: int
        Number relating to how many times the GP has been evaluated
    STEP: int
        Relates to window size, if TIME_STEPS == STEP, there is no window, but one can create
        overlapping segments if the STEP size is changed.

    Returns
    -------
    (Xs, ys): (np.ndarray, np.ndarray):
        A matrix of X values with corresponding y labels given as a column vector

    Examples
    --------
    >>> cols = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
    >>> robust_scale(df, cols)
    >>> TIME_STEPS = timesteps
    >>> STEP = step
    >>> Xs, ys = create_dataset(df[cols], df.target, TIME_STEPS, STEP)
    >>> X_train, X_test, y_train, y_test = model_selection.train_test_split(
    ...     Xs, ys, random_state=RANDOM_SEED
    ... )
    """

    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i : (i + time_steps)].values
        labels = y.iloc[i : i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])

    return np.array(Xs), np.array(ys).reshape(-1, 1)


def get_encoding(
    dataset: str, dataform: str = None
) -> (OneHotEncoder, List[str], List[str]):
    """Get inverse of the OneHotEncoder used intially encoding the original labels"""

    if dataform is not None:
        encoding_filename = f"{asnwd}/data/{dataform}-{dataset}.encoding"
    else:
        encoding_filename = f"{asnwd}/data/{dataset}.encoding"

    with open(encoding_filename, "rb") as eb:
        encoding = joblib.load(eb)
    class_encoding = encoding.categories_[0]

    if dataset == "plasticc":
        class_mapping = PLASTICC_CLASS_MAPPING
        class_names = list(np.vectorize(class_mapping.get)(class_encoding))
    else:
        class_names = class_encoding

    return encoding, class_encoding, class_names


def load_wisdm_2010(timesteps=200, step=200):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Load WISDM-2010 dataset
    column_names = [
        "user_id",
        "activity",
        "timestamp",
        "x_axis",
        "y_axis",
        "z_axis",
    ]

    df = pd.read_csv(
        f"{Path(__file__).absolute().parent.parent}/data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt",
        header=None,
        names=column_names,
    )

    df.z_axis.replace(regex=True, inplace=True, to_replace=r";", value=r"")
    df["z_axis"] = df.z_axis.astype(np.float64)
    df.dropna(axis=0, how="any", inplace=True)

    cols = ["x_axis", "y_axis", "z_axis"]

    # Perfrom robust scaling
    robust_scale(df, cols)

    TIME_STEPS = timesteps
    STEP = step

    Xs, ys = create_dataset(df[cols], df.activity, TIME_STEPS, STEP)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        Xs, ys, random_state=RANDOM_SEED
    )

    return X_train, y_train, X_test, y_test


def load_wisdm_2019(timesteps=200, step=200):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Load WISDM-2019 dataset
    # column_names = [
    #     "phone_accel_x",
    #     "phone_accel_y",
    #     "phone_accel_z",
    #     "phone_gyro_x",
    #     "phone_gyro_y",
    #     "phone_gyro_z",
    # ]

    with open(f"{asnwd}/data/wisdm-dataset/activity_key.txt") as f:
        activity_list = f.read().split("\n")

    # Use activity_map.values() for activity names when plotting, but then the confusion matrix will
    # be squashed
    activity_map = {}
    for element in activity_list:
        split = element.split(" = ")
        if len(split) < 2:
            continue
        activity_map[split[1]] = split[0]

    # As this is quite a large dataset, to save pre-processing time, I have taken "ready-made"
    # dataframe files from: https://github.com/LACoderDeBH/CS230_HAR_WISDM

    # The work presented there and published by Susana Benavidez et al 2019 is what will be used to
    # compare results with laatest attempts at applying deep learning methods to the updated WISDM
    # dataset
    with open(f"{asnwd}/data/wisdm-dataset/phone.df", "rb") as phone_dataframe:
        df = pickle.load(phone_dataframe)

    assert df.shape == (4780251, 9)

    cols = [
        "phone_accel_x",
        "phone_accel_y",
        "phone_accel_z",
    ]

    robust_scale(df, cols)

    TIME_STEPS = timesteps
    STEP = step

    Xs, ys = create_dataset(df[cols], df.activity, TIME_STEPS, STEP)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        Xs, ys, random_state=RANDOM_SEED
    )

    return X_train, y_train, X_test, y_test


def load_mts(dataset):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    X_train = np.load(f"{asnwd}/data/transformed-mtsdata/{dataset}/x_train.npy")
    y_train = np.load(f"{asnwd}/data/transformed-mtsdata/{dataset}/y_train.npy")
    X_test = np.load(f"{asnwd}/data/transformed-mtsdata/{dataset}/x_test.npy")
    y_test = np.load(f"{asnwd}/data/transformed-mtsdata/{dataset}/y_test.npy")

    return X_train, y_train, X_test, y_test


def text_to_bits(text, encoding="utf-8", errors="surrogatepass"):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """
    bits = bin(int.from_bytes(text.encode(encoding, errors), "big"))[2:]
    return int(bits.zfill(8 * ((len(bits) + 7) // 8)), 2)


def text_from_bits(bits, encoding="utf-8", errors="surrogatepass"):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """
    bits = bin(bits)
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, "big").decode(encoding, errors) or "\0"


def __load_plasticc_dataset_from_csv(timesteps, snonly=None):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    data = pd.read_csv(
        f"{asnwd}/data/plasticc/training_set.csv",
        sep=",",
    )
    data = remap_filters(df=data, filter_map=LSST_FILTER_MAP)
    data.rename(
        {"flux_err": "flux_error"}, axis="columns", inplace=True
    )  # snmachine and PLAsTiCC uses a different denomination

    if snonly is not None:
        dataform = "snonly"
        df = __filter_dataframe_only_supernova(
            f"{asnwd}/data/plasticc/train_subset.txt",
            data,
        )
    else:
        dataform = "full"
        df = data

    object_list = list(np.unique(df["object_id"]))

    obs_transient, _ = __transient_trim(object_list, df)
    generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
    )
    generated_gp_dataset["object_id"] = generated_gp_dataset["object_id"].astype(int)

    metadata_pd = pd.read_csv(
        f"{asnwd}/data/plasticc/training_set_metadata.csv",
        sep=",",
        index_col="object_id",
    )

    metadata_pd = metadata_pd.reset_index()
    metadata_pd["object_id"] = metadata_pd["object_id"].astype(int)

    df_with_labels = generated_gp_dataset.merge(metadata_pd, on="object_id", how="left")

    df = df_with_labels.filter(
        items=[
            "mjd",
            "lsstg",
            "lssti",
            "lsstr",
            "lsstu",
            "lssty",
            "lsstz",
            "object_id",
            "hostgal_photoz",
            "hostgal_photoz_err",
            "target",
        ]
    )

    print(df.dtypes)
    df.convert_dtypes()
    df["object_id"] = df["object_id"].astype(int)
    print(df.dtypes)

    print(df.columns)
    print(df.head())
    print(df.dtypes)

    df.to_csv(
        f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_with_z.csv"
    )

    # df.to_parquet(
    #     f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_with_z.parquet",
    #     engine="pyarrow",
    #     compression="snappy",
    # )

    return df


def __load_plasticc_test_set_dataset_from_csv(
    timesteps, snonly=None, batch_filename=None
):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # batch_filename = "plasticc_test_lightcurves_01"

    data = pd.read_csv(
        f"{asnwd}/data/plasticc/test_set/{batch_filename}.csv",
        sep=",",
    )
    data = remap_filters(df=data, filter_map=LSST_FILTER_MAP)
    data.rename(
        {"flux_err": "flux_error"}, axis="columns", inplace=True
    )  # snmachine and PLAsTiCC uses a different denomination
    data.rename(
        {"detected_bool": "detected"}, axis="columns", inplace=True
    )  # snmachine and PLAsTiCC uses a different denomination

    if snonly is not None:
        dataform = "snonly"
        df = __filter_dataframe_only_supernova(
            f"{asnwd}/data/plasticc/train_subset.txt",
            data,
        )
    else:
        dataform = "full_test"
        df = data

    object_list = list(np.unique(df["object_id"]))

    obs_transient = __transient_trim(object_list, df)
    generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
    )
    generated_gp_dataset["object_id"] = generated_gp_dataset["object_id"].astype(int)

    metadata_pd = pd.read_csv(
        f"{asnwd}/data/plasticc/test_set/plasticc_test_metadata.csv",
        sep=",",
        index_col="object_id",
    )

    metadata_pd = metadata_pd.reset_index()
    metadata_pd["object_id"] = metadata_pd["object_id"].astype(int)

    df_with_labels = generated_gp_dataset.merge(metadata_pd, on="object_id", how="left")

    df = df_with_labels.filter(
        items=[
            "mjd",
            "lsstg",
            "lssti",
            "lsstr",
            "lsstu",
            "lssty",
            "lsstz",
            "object_id",
            "hostgal_photoz",
            "hostgal_photoz_err",
            "true_target",
        ]
    )

    print(df.dtypes)
    df.convert_dtypes()
    df["object_id"] = df["object_id"].astype(int)
    print(df.dtypes)

    print(df.columns)
    print(df.head())
    print(df.dtypes)

    df.to_csv(
        f"{asnwd}/data/plasticc/test_set/{dataform}_transformed_df_timesteps_{timesteps}_with_z_{batch_filename}.csv"
    )

    # df.to_parquet(
    #     f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_with_z.parquet",
    #     engine="pyarrow",
    #     compression="snappy",
    # )

    return df


def __load_avocado_plasticc_dataset_from_csv(
    timesteps, snonly=None, batch_filename=None
):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    data = pd.read_csv(
        f"{asnwd}/data/plasticc/avocado/{batch_filename}.csv",
        sep=",",
    )

    data.rename({"band": "filter"}, axis="columns", inplace=True)
    data.rename({"time": "mjd"}, axis="columns", inplace=True)

    if snonly is not None:
        dataform = "snonly"
        df = __filter_dataframe_only_supernova(
            f"{asnwd}/data/plasticc/train_subset.txt",
            data,
        )
    else:
        dataform = "avocado"
        df = data

    object_list = list(np.unique(df["object_id"]))

    obs_transient, object_list = __transient_trim(object_list, df)
    generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
    )
    # generated_gp_dataset['object_id'] = generated_gp_dataset['object_id'].astype(int)

    metadata_pd = pd.read_csv(
        f"{asnwd}/data/plasticc/avocado/avo_aug_0.csv",
        sep=",",
        index_col="object_id",
    )

    metadata_pd = metadata_pd.reset_index()
    # metadata_pd['object_id'] = metadata_pd['object_id'].astype(int)

    df_with_labels = generated_gp_dataset.merge(metadata_pd, on="object_id", how="left")

    df_with_labels.rename({"class": "target"}, axis="columns", inplace=True)
    df_with_labels.rename(
        {"host_photoz": "hostgal_photoz"}, axis="columns", inplace=True
    )
    df_with_labels.rename(
        {"host_photoz_error": "hostgal_photoz_err"}, axis="columns", inplace=True
    )

    df = df_with_labels.filter(
        items=[
            "mjd",
            "lsstg",
            "lssti",
            "lsstr",
            "lsstu",
            "lssty",
            "lsstz",
            "object_id",
            "hostgal_photoz",
            "hostgal_photoz_err",
            "target",
        ]
    )

    print(df.dtypes)
    df.convert_dtypes()
    # df['object_id'] = df['object_id'].astype(int)
    print(df.dtypes)

    print(df.columns)
    print(df.head())
    print(df.dtypes)

    df.to_csv(
        f"{asnwd}/data/plasticc/avocado/{dataform}_transformed_df_timesteps_{timesteps}_with_z_{batch_filename}.csv"
    )

    return df


def __generate_augmented_plasticc_dataset_from_pickle(augmented_binary):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    # augmented_binary = f"{asnwd}/data/plasticc/aug_z_new_long_many_obs_35k.pckl"

    with open(augmented_binary, "rb") as f:
        aug = pickle.load(f)

    aug.metadata.to_csv(
        f"{asnwd}/data/plasticc/augmented_training_set_metadata.csv",
    )

    object_list = list(aug.object_names)
    cols = list(aug.data[object_list[0]].to_pandas().columns)

    adf = pd.DataFrame(
        data=[],
        columns=cols,
    )

    for i in range(len(object_list)):
        adf = pd.concat([adf, aug.data[object_list[i]].to_pandas()])

    # adf = adf.set_index('object_id')
    adf = adf.replace({"_aug": "000"}, regex=True)
    adf = adf.convert_dtypes()
    print(adf.dtypes)
    print(adf.head())
    print(adf["object_id"].values)
    np.savetxt(
        f"{asnwd}/data/plasticc/aug_object_list.txt", adf["object_id"].values, fmt="%s"
    )

    # try:
    #     adf.to_parquet(
    #         f"{asnwd}/data/plasticc/augmented_training_set.parquet",
    #         engine="pyarrow",
    #         compression="snappy",
    #     )
    # except IOError:
    adf.to_csv(f"{asnwd}/data/plasticc/augmented_training_set.csv")

    return adf


def __load_augmented_plasticc_dataset_from_csv(timesteps):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    try:
        data = pd.read_csv(
            f"{asnwd}/data/plasticc/augmented_training_set.csv",
        )
    except IOError:
        data = __generate_augmented_plasticc_dataset_from_pickle(
            f"{asnwd}/data/plasticc/aug_z_new_long_46k.pckl"
        )

    data = data.replace({"_aug": "000"}, regex=True)
    data = data.convert_dtypes()

    # data = __remap_filters(df=data)
    # data.rename(
    #     {"flux_err": "flux_error"}, axis="columns", inplace=True
    # )  # snmachine and PLAsTiCC uses a different denomination

    print(data.head())

    df = __filter_dataframe_only_supernova(
        f"{asnwd}/data/plasticc/aug_object_list.txt",
        data,
    )

    print(df.head())
    print(df.dtypes)

    object_list = list(np.unique(df["object_id"]))
    print(len(object_list))

    # obs_transient = __transient_trim(object_list, df)
    generated_gp_dataset = generate_gp_all_objects(
        object_list, df, timesteps, LSST_PB_WAVELENGTHS
    )
    generated_gp_dataset["object_id"] = generated_gp_dataset["object_id"].astype(int)

    metadata_pd = pd.read_csv(
        f"{asnwd}/data/plasticc/augmented_training_set_metadata.csv",
        sep=",",
        index_col="object_id",
    )

    metadata_pd = metadata_pd.reset_index()
    metadata_pd = metadata_pd.replace({"_aug": "000"}, regex=True)
    metadata_pd = metadata_pd.convert_dtypes()
    metadata_pd["object_id"] = metadata_pd["object_id"].astype(int)

    df_with_labels = generated_gp_dataset.merge(metadata_pd, on="object_id", how="left")
    print(df_with_labels.columns)
    print(df_with_labels.head())
    print(df_with_labels.dtypes)

    df = df_with_labels.filter(
        items=[
            "mjd",
            "lsstg",
            "lssti",
            "lsstr",
            "lsstu",
            "lssty",
            "lsstz",
            "object_id",
            "hostgal_photoz",
            "hostgal_photoz_err",
            "target",
        ]
    )

    print(df.dtypes)
    df.convert_dtypes()
    df["object_id"] = df["object_id"].astype(int)
    print(df.dtypes)

    print(df.columns)
    print(df.head())
    print(df.dtypes)

    # df.to_csv(f"{asnwd}/data/plasticc/augmented_transformed_df_timesteps_{timesteps}_with_z_backup.csv")
    df.to_csv(
        f"{asnwd}/data/plasticc/augmented_transformed_df_timesteps_{timesteps}_with_z.csv"
    )

    # try:
    #     df.to_parquet(
    #         f"{asnwd}/data/plasticc/augmented_transformed_df_timesteps_{timesteps}_with_z.parquet",
    #         engine="pyarrow",
    #         compression="snappy",
    #     )

    # except IOError:
    #     df.to_csv(f"{asnwd}/data/plasticc/augmented_transformed_df_timesteps_{timesteps}_with_z.csv")

    return df


def get_data_count(dataset, y_train, y_test, dataform=None):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    if dataform is not None:
        dataform = "testset"
        with open(f"{asnwd}/data/{dataform}-{dataset}.encoding", "rb") as eb:
            encoding = joblib.load(eb)
    else:
        print("Using WISDM or MTS data")
        with open(f"{asnwd}/data/{dataset}.encoding", "rb") as eb:
            encoding = joblib.load(eb)

    from collections import Counter

    from pandas.core.common import flatten

    y_true_train = encoding.inverse_transform(y_train)
    y_train_count = Counter(list(flatten(y_true_train)))
    print("N_TRAIN:", y_train_count)

    y_true_test = encoding.inverse_transform(y_test)
    y_test_count = Counter(list(flatten(y_true_test)))
    print("N_TEST:", y_test_count)

    return y_train_count, y_test_count


def load_plasticc(
    timesteps=100, step=100, redshift=None, augmented=None, snonly=None, avocado=None
):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    TIME_STEPS = timesteps
    STEP = step

    if augmented is not None:
        dataform = "augmented"
        try:
            df = pd.read_csv(
                f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_with_z.csv",
                sep=",",
            )
            # df = pd.read_parquet(
            #     f"{asnwd}/data/plasticc/augmented_transformed_df_timesteps_{timesteps}_with_z.parquet",
            #     engine="pyarrow",
            # )

        except IOError:
            df = __load_augmented_plasticc_dataset_from_csv(timesteps)
    elif snonly is not None:
        dataform = "snonly"
        try:
            df = pd.read_csv(
                f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_with_z.csv",
            )

        except IOError:
            df = __load_plasticc_dataset_from_csv(timesteps, snonly=True)
    elif avocado is not None:
        dataform = "avocado"
        try:
            df = pd.read_csv(
                f"{asnwd}/data/plasticc/avocado/{dataform}_transformed_df_timesteps_{timesteps}_with_z.csv",
            )

        except IOError:
            df = __load_avocado_plasticc_dataset_from_csv(timesteps)
    else:
        dataform = "full"
        try:
            df = pd.read_csv(
                f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_with_z.csv",
            )

        except IOError:
            df = __load_plasticc_dataset_from_csv(timesteps)

    cols = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
    robust_scale(df, cols)

    Xs, ys = create_dataset(df[cols], df.target, TIME_STEPS, STEP)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        Xs, ys, random_state=RANDOM_SEED
    )

    np.save(
        f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_X_train.npy",
        X_train,
    )
    np.save(
        f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_X_test.npy",
        X_test,
    )
    np.save(
        f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_y_train.npy",
        y_train,
    )
    np.save(
        f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_y_test.npy",
        y_test,
    )

    if redshift is None:
        return X_train, y_train, X_test, y_test
    else:
        zcols = ["hostgal_photoz", "hostgal_photoz_err"]
        robust_scale(df, zcols)

        ZXs, zys = create_dataset(df[zcols], df.target, TIME_STEPS, STEP)

        ZX = []
        for z in range(0, len(ZXs)):
            ZX.append(stats.mode(ZXs[z])[0][0])

        ZX_train, ZX_test, _, _ = model_selection.train_test_split(
            np.array(ZX), zys, random_state=RANDOM_SEED
        )

        np.save(
            f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_ZX_train.npy",
            ZX_train,
        )
        np.save(
            f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_ZX_test.npy",
            ZX_test,
        )

        return X_train, y_train, X_test, y_test, ZX_train, ZX_test


def load_full_avocado_plasticc_from_numpy(
    timesteps=100, redshift=None, batch_filename=None
):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    # dataform = "avocado"
    try:
        X_train = np.load(
            f"{asnwd}/data/plasticc/avocado/avocado__transformed_df_timesteps_100_X_full_avo_train.npy",
        )

        y_train = np.load(
            f"{asnwd}/data/plasticc/avocado/avocado__transformed_df_timesteps_100_y_full_avo_train.npy",
        )

        Z_train = np.load(
            f"{asnwd}/data/plasticc/avocado/avocado__transformed_df_timesteps_100_Z_full_avo_train.npy",
        )

        X_full_test_no_99 = np.load(
            f"{asnwd}/data/plasticc/test_set/full_test_transformed_df_timesteps_100_X_full_test_no_99.npy",
            # mmap_mode='r'
        )

        y_full_test_no_99 = np.load(
            f"{asnwd}/data/plasticc/test_set/full_test_transformed_df_timesteps_100_y_full_test_no_99.npy",
        )

        Z_full_test_no_99 = np.load(
            f"{asnwd}/data/plasticc/test_set/full_test_transformed_df_timesteps_100_Z_full_test_no_99.npy",
        )

    except IOError:
        X_train, y_train, Z_train = save_avocado_training_set(
            batch_filename=batch_filename
        )

    if redshift is not None:
        return (
            X_train,
            y_train,
            X_full_test_no_99,
            y_full_test_no_99,
            Z_train,
            Z_full_test_no_99,
        )
    else:
        return X_train, y_train


def load_full_plasticc_test_from_numpy(timesteps=100, redshift=None):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    try:
        X_full_test_no_99 = np.load(
            f"{asnwd}/data/plasticc/test_set/no99/full_test_transformed_df_timesteps_100_X_full_test_no_99.npy",
            # mmap_mode='r'
        )

        y_full_test_no_99 = np.load(
            f"{asnwd}/data/plasticc/test_set/no99/full_test_transformed_df_timesteps_100_y_full_test_no_99.npy",
        )

        Z_full_test_no_99 = np.load(
            f"{asnwd}/data/plasticc/test_set/no99/full_test_transformed_df_timesteps_100_Z_full_test_no_99.npy",
        )

    except IOError:
        X_full_test = np.load(
            f"{asnwd}/data/plasticc/test_set/full_test_transformed_df_timesteps_100_X_full_test.npy",
            # mmap_mode='r'
        )

        y_full_test = np.load(
            f"{asnwd}/data/plasticc/test_set/full_test_transformed_df_timesteps_100_y_full_test.npy",
        )

        Z_full_test = np.load(
            f"{asnwd}/data/plasticc/test_set/full_test_transformed_df_timesteps_100_Z_full_test.npy",
        )

        print(X_full_test.shape, y_full_test.shape, Z_full_test.shape)

        # Get index of class 99, append index of those NOT 99 to 'keep' list
        class_99_index = []
        for i in range(len(y_full_test.flatten())):
            if y_full_test.flatten()[i] in [991, 992, 993, 994]:
                continue
            else:
                class_99_index.append(i)

        print(len(class_99_index))

        filter_indices = class_99_index
        axis = 0
        array = X_full_test
        arrayY = y_full_test
        arrayZ = Z_full_test

        X_full_test_no_99 = np.take(array, filter_indices, axis)
        y_full_test_no_99 = np.take(arrayY, filter_indices, axis)
        Z_full_test_no_99 = np.take(arrayZ, filter_indices, axis)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_full_test_no_99, y_full_test_no_99, train_size=0.75, random_state=RANDOM_SEED
    )

    Z_train, Z_test, _, _ = model_selection.train_test_split(
        Z_full_test_no_99, y_full_test_no_99, train_size=0.75, random_state=RANDOM_SEED
    )

    if redshift is not None:
        return X_train, y_train, X_test, y_test, Z_train, Z_test
    else:
        return X_train, y_train, X_test, y_test


def save_avocado_training_set(
    timesteps=100,
    step=100,
    redshift=None,
    augmented=None,
    snonly=None,
    batch_filename=None,
):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    TIME_STEPS = timesteps
    STEP = step

    dataform = "avocado"
    try:
        df = pd.read_csv(
            f"{asnwd}/data/plasticc/avocado/{dataform}_transformed_df_timesteps_{timesteps}_with_z.csv",
        )

    except IOError:
        df = __load_avocado_plasticc_dataset_from_csv(
            timesteps, batch_filename=batch_filename
        )

    cols = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
    robust_scale(df, cols)

    Xs, ys = create_dataset(df[cols], df.target, TIME_STEPS, STEP)

    # X_train, X_test, y_train, y_test = model_selection.train_test_split(
    #     Xs, ys, random_state=RANDOM_SEED
    # )

    np.save(
        f"{asnwd}/data/plasticc/avocado/{dataform}_transformed_df_timesteps_{timesteps}_X_train_{batch_filename}.npy",
        Xs,
    )
    np.save(
        f"{asnwd}/data/plasticc/avocado/{dataform}_transformed_df_timesteps_{timesteps}_y_train_{batch_filename}.npy",
        ys,
    )

    if redshift is None:
        return Xs, ys
    else:
        zcols = ["hostgal_photoz", "hostgal_photoz_err"]
        robust_scale(df, zcols)

        ZXs, zys = create_dataset(df[zcols], df.target, TIME_STEPS, STEP)

        ZX = []
        for z in range(0, len(ZXs)):
            ZX.append(stats.mode(ZXs[z])[0][0])

        # ZX_train, ZX_test, _, _ = model_selection.train_test_split(
        #     np.array(ZX), zys, random_state=RANDOM_SEED
        # )

        np.save(
            f"{asnwd}/data/plasticc/avocado/{dataform}_transformed_df_timesteps_{timesteps}_Z_train_{batch_filename}.npy",
            np.array(ZX),
        )
        # np.save(
        #         f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_ZX_test.npy",
        #         ZX_test,
        # )

        return Xs, ys, np.array(ZX)


def save_plasticc_test_set(
    timesteps=100,
    step=100,
    redshift=None,
    augmented=None,
    snonly=None,
    batch_filename=None,
):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    TIME_STEPS = timesteps
    STEP = step

    if augmented is not None:
        dataform = "augmented"
        try:
            df = pd.read_csv(
                f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_with_z.csv",
                sep=",",
            )
            # df = pd.read_parquet(
            #     f"{asnwd}/data/plasticc/augmented_transformed_df_timesteps_{timesteps}_with_z.parquet",
            #     engine="pyarrow",
            # )

        except IOError:
            df = __load_augmented_plasticc_dataset_from_csv(timesteps)
    elif snonly is not None:
        dataform = "snonly"
        try:
            df = pd.read_csv(
                f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_with_z.csv",
            )

        except IOError:
            df = __load_plasticc_dataset_from_csv(timesteps, snonly=True)
    else:
        dataform = "full_test"
        try:
            df = pd.read_csv(
                f"{asnwd}/data/plasticc/test_set/{dataform}_transformed_df_timesteps_{timesteps}_with_z.csv",
            )

        except IOError:
            df = __load_plasticc_test_set_dataset_from_csv(
                timesteps, batch_filename=batch_filename
            )

    cols = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
    robust_scale(df, cols)

    Xs, ys = create_dataset(df[cols], df.true_target, TIME_STEPS, STEP)

    # X_train, X_test, y_train, y_test = model_selection.train_test_split(
    #     Xs, ys, random_state=RANDOM_SEED
    # )

    np.save(
        f"{asnwd}/data/plasticc/test_set/{dataform}_transformed_df_timesteps_{timesteps}_X_test_{batch_filename}.npy",
        Xs,
    )
    np.save(
        f"{asnwd}/data/plasticc/test_set/{dataform}_transformed_df_timesteps_{timesteps}_y_test_{batch_filename}.npy",
        ys,
    )

    if redshift is None:
        return Xs, ys
    else:
        zcols = ["hostgal_photoz", "hostgal_photoz_err"]
        robust_scale(df, zcols)

        ZXs, zys = create_dataset(df[zcols], df.true_target, TIME_STEPS, STEP)

        ZX = []
        for z in range(0, len(ZXs)):
            ZX.append(stats.mode(ZXs[z])[0][0])

        # ZX_train, ZX_test, _, _ = model_selection.train_test_split(
        #     np.array(ZX), zys, random_state=RANDOM_SEED
        # )

        np.save(
            f"{asnwd}/data/plasticc/test_set/{dataform}_transformed_df_timesteps_{timesteps}_ZX_test_{batch_filename}.npy",
            np.array(ZX),
        )
        # np.save(
        #         f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_ZX_test.npy",
        #         ZX_test,
        # )

        return Xs, ys, np.array(ZX)


def load_dataset(
    dataset,
    redshift=None,
    balance=None,
    augmented=None,
    snonly=None,
    avocado=None,
    testset=None,
    fink=None,
):
    # TODO: Update docstrings
    """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    df: pd.DataFrame
        DataFrame containing the full light curve including dead points.

    Returns
    -------
    obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
        Tuple containing the updated dataframe with only the transient section, and a list of
        objects that the transformation was successful for. Note, some objects may cause an error
        and hence would not be returned in the new transformed dataframe

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """
    if dataset == "wisdm_2010":
        dataform = None
        # Load data
        X_train, y_train, X_test, y_test = load_wisdm_2010()
        # One hot encode y
        enc, y_train, y_test = one_hot_encode(y_train, y_test)
        encoding_file = f"{asnwd}/data/{dataset}.encoding"
        if not os.path.exists(encoding_file):
            with open(encoding_file, "wb") as f:
                joblib.dump(enc, f)

        loss = "categorical_crossentropy"

    elif dataset == "wisdm_2019":
        dataform = None
        # Load data
        X_train, y_train, X_test, y_test = load_wisdm_2019()
        # One hot encode y
        enc, y_train, y_test = one_hot_encode(y_train, y_test)
        encoding_file = f"{asnwd}/data/{dataset}.encoding"
        if not os.path.exists(encoding_file):
            with open(encoding_file, "wb") as f:
                joblib.dump(enc, f)

        loss = "categorical_crossentropy"

    elif dataset in [
        "ArabicDigits",
        "AUSLAN",
        "CharacterTrajectories",
        "CMUsubject16",
        "ECG",
        "JapaneseVowels",
        "KickvsPunch",
        "Libras",
        "NetFlow",
        "UWave",
        "Wafer",
        "WalkvsRun",
    ]:
        dataform = None
        # Load data
        X_train, y_train, X_test, y_test = load_mts(dataset)
        # transform the labels from integers to one hot vectors
        import sklearn

        enc = sklearn.preprocessing.OneHotEncoder(categories="auto")
        enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
        encoding_file = f"{asnwd}/data/{dataset}.encoding"
        if not os.path.exists(encoding_file):
            with open(encoding_file, "wb") as f:
                joblib.dump(enc, f)

        # save orignal y because later we will use binary
        # y_true = np.argmax(y_test, axis=1)

        if len(X_train.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        loss = "categorical_crossentropy"

    elif dataset == "plasticc":
        # Load data
        if redshift is None:
            if avocado is not None:
                (
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                ) = load_full_avocado_plasticc_from_numpy(redshift=redshift)
            elif testset is not None:
                X_train, y_train, X_test, y_test = load_full_plasticc_test_from_numpy(
                    redshift=redshift
                )
            else:
                X_train, y_train, X_test, y_test = load_plasticc(
                    augmented=augmented, snonly=snonly, avocado=avocado
                )
        else:  # With redshift
            if testset is not None:
                (
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    ZX_train,
                    ZX_test,
                ) = load_full_plasticc_test_from_numpy(redshift=redshift)
            elif avocado is not None:
                (
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    ZX_train,
                    ZX_test,
                ) = load_full_avocado_plasticc_from_numpy(redshift=redshift)
            else:
                X_train, y_train, X_test, y_test, ZX_train, ZX_test = load_plasticc(
                    redshift=redshift,
                    augmented=augmented,
                    snonly=snonly,
                    avocado=avocado,
                )

        if augmented is not None:
            dataform = "augmented"
        elif snonly is not None:
            dataform = "snonly"
        elif avocado is not None:
            dataform = "avocado"
        elif testset is not None:
            dataform = "testset"
        else:
            dataform = "full"
        # One hot encode y
        enc, y_train, y_test = one_hot_encode(y_train, y_test)
        encoding_file = f"{asnwd}/data/{dataform}-{dataset}.encoding"
        if not os.path.exists(encoding_file):
            with open(encoding_file, "wb") as f:
                joblib.dump(enc, f)

        loss = WeightedLogLoss()

    if balance is not None:
        (
            num_samples,
            timesteps,
            num_features,
        ) = X_train.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)

        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        # random_state: if None, the random number generator is the RandomState instance used by np.random.

        from imblearn.over_sampling import SVMSMOTE
        from imblearn.under_sampling import (
            InstanceHardnessThreshold,
            RandomUnderSampler,
        )

        # sampler = SVMSMOTE(sampling_strategy="not majority")
        # sampler = InstanceHardnessThreshold(sampling_strategy="not minority")
        sampler = RandomUnderSampler(sampling_strategy="not minority")

        X_resampled, y_resampled = sampler.fit_resample(
            X_train.reshape(X_train.shape[0], -1), y_train
        )

        # Re-shape 2D data back to 3D original shape, i.e (BATCH_SIZE, timesteps, num_features)
        X_resampled = np.reshape(
            X_resampled, (X_resampled.shape[0], timesteps, num_features)
        )

        if redshift is not None:
            num_z_samples, num_z_features = ZX_train.shape
            Z_resampled, _ = sampler.fit_resample(ZX_train, y_train)
            Z_resampled = np.reshape(
                Z_resampled, (Z_resampled.shape[0], num_z_features)
            )

            ZX_train = Z_resampled

        X_train = X_resampled
        y_train = y_resampled

    if dataform is not None:
        y_train_count, y_test_count = get_data_count(
            dataset, y_train, y_test, dataform=dataform
        )

    if redshift is None:
        if fink is not None:
            X_train = X_train[:, :, 0:3:2]
            X_test = X_test[:, :, 0:3:2]
            return X_train, y_train, X_test, y_test, loss
        return X_train, y_train, X_test, y_test, loss
    else:
        return X_train, y_train, X_test, y_test, loss, ZX_train, ZX_test
