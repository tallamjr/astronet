from sklearn.metrics import confusion_matrix
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import logging

from pathlib import Path
# print("File      Path:", Path(__file__).absolute())
# print("Parent of Directory Path:", Path().absolute().parent)
from astronet.t2.preprocess import robust_scale


def t2_logger(name, level="INFO"):
    """ Initialise python logger.

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

    # Create formatter
    # Format of the log message to be printed
    FORMAT = "[%(asctime)s] "
    FORMAT += "{%(filename)s:%(lineno)d} "
    FORMAT += "%(levelname)s "
    FORMAT += "- %(message)s"
    # FORMAT = "[%(asctime)-15s]"
    # FORMAT += "[%(filename)s:"
    # FORMAT += "%(lineno)s] -- "
    # FORMAT += "%(levelname)s "
    # FORMAT += "%(message)s"
    # Date format
    DATEFORMAT = "%y-%m-%d.%H:%M:%S"

    formatter = logging.Formatter(fmt=FORMAT, datefmt=DATEFORMAT)
    # Add formatter to ch
    ch.setFormatter(formatter)
    # Add ch to logger
    logger.addHandler(ch)

    return logger


def train_val_test_split(df, cols):

    features = df[cols]
    # column_indices = {name: i for i, name in enumerate(features.columns)}

    n = len(df)
    df_train = df[0 : int(n * 0.8)].copy()
    df_val = df[int(n * 0.8) : int(n * 0.95)].copy()
    df_test = df[int(n * 0.95) :].copy()

    num_features = features.shape[1]

    return df_train, df_val, df_test, num_features


def plot_activity(activity, df, cols):
    data = df[df["activity"] == activity][cols][:400]
    axis = data.plot(subplots=True, figsize=(16, 12), title=activity)
    for ax in axis:
        ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.5))


def create_dataset(X, y, time_steps=1, step=1):
    from scipy import stats
    import numpy as np

    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i : (i + time_steps)].values
        labels = y.iloc[i : i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])

    return np.array(Xs), np.array(ys).reshape(-1, 1)


def load_WISDM(timesteps=200, step=40):

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Load WISDM-2010 or WISDM-2019 dataset
    column_names = [
        "user_id",
        "activity",
        "timestamp",
        "x_axis",
        "y_axis",
        "z_axis",
    ]

    df = pd.read_csv(str(Path(__file__).absolute().parent.parent.parent) +
        "/data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt",
        header=None,
        names=column_names,
    )
    df.z_axis.replace(regex=True, inplace=True, to_replace=r";", value=r"")
    df["z_axis"] = df.z_axis.astype(np.float64)
    df.dropna(axis=0, how="any", inplace=True)

    # print(df.head())

    cols = ["x_axis", "y_axis", "z_axis"]

    # print(df[cols].head())

    df_train, df_val, df_test, num_features = train_val_test_split(df, cols)
    # print(num_features)  # Should = 3 in this case

    # Perfrom robust scaling
    robust_scale(df_train, df_val, df_test, cols)

    TIME_STEPS = timesteps
    STEP = step

    X_train, y_train = create_dataset(
        df_train[cols],
        df_train.activity,
        TIME_STEPS,
        STEP
    )

    X_val, y_val = create_dataset(
        df_val[cols],
        df_val.activity,
        TIME_STEPS,
        STEP
    )

    X_test, y_test = create_dataset(
        df_test[cols],
        df_test.activity,
        TIME_STEPS,
        STEP
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
