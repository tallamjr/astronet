import logging
import numpy as np
import pandas as pd
import pickle
import sys
import tensorflow as tf

from pathlib import Path

from astronet.t2.constants import pb_wavelengths, astronet_working_directory as asnwd
from astronet.t2.preprocess import robust_scale, fit_2d_gp, predict_2d_gp

# 'SettingWithCopyWarning' in Pandas: https://bit.ly/3mv3fhw
pd.options.mode.chained_assignment = None  # default='warn'


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
    # Date format
    DATEFORMAT = "%y-%m-%d %H:%M:%S"

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


def load_wisdm_2010(timesteps=200, step=200):

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
        str(Path(__file__).absolute().parent.parent.parent) +
        "/data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt",
        header=None,
        names=column_names,
    )

    df.z_axis.replace(regex=True, inplace=True, to_replace=r";", value=r"")
    df["z_axis"] = df.z_axis.astype(np.float64)
    df.dropna(axis=0, how="any", inplace=True)

    cols = ["x_axis", "y_axis", "z_axis"]

    df_train, df_val, df_test, num_features = train_val_test_split(df, cols)
    assert num_features == 3  # Should = 3 in this case

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


#  TODO:
# Investigate performance of timesteps=200 for wisdm_2019 since this is the timesteps used for
# widsm_2010 which obtains better performance.
def load_wisdm_2019(timesteps=200, step=200):

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
    with open(f"{asnwd}/data/wisdm-dataset/phone.df", "rb") as df:
            phone = pickle.load(df)

    assert phone.shape == (4780251, 9)

    cols = [
        "phone_accel_x",
        "phone_accel_y",
        "phone_accel_z",
    ]

    df_train, df_val, df_test, num_features = train_val_test_split(phone, cols)
    assert num_features == 3

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

    # assert y_train.shape == (95603, 1)

    return X_train, y_train, X_val, y_val, X_test, y_test


def __remap_filters(df):
    """Function to remap integer filters to the corresponding lsst filters and
    also to set filter name syntax to what snmachine already recognizes

    df: pandas.dataframe
        Dataframe of lightcurve observations
    """
    df.rename({'passband': 'filter'}, axis='columns', inplace=True)
    filter_replace = {0: 'lsstu', 1: 'lsstg', 2: 'lsstr', 3: 'lssti',
                      4: 'lsstz', 5: 'lssty'}
    df['filter'].replace(to_replace=filter_replace, inplace=True)
    return df


def __filter_dataframe_only_supernova(object_list_filename, dataframe):

    plasticc_object_list = np.genfromtxt(object_list_filename, dtype='U')
    filtered_dataframe = dataframe[dataframe['object_id'].isin(plasticc_object_list)]
    return filtered_dataframe


def __transient_trim(object_list, df):
    adf = pd.DataFrame(data=[], columns=df.columns)
    for obj in object_list:
        obs = df[df['object_id'] == obj]
        obs_time = obs['mjd']
        obs_detected_time = obs_time[obs['detected'] == 1]
        is_obs_transient = (obs_time > obs_detected_time.iat[0] - 50) & (obs_time < obs_detected_time.iat[-1] + 50)
        obs_transient = obs[is_obs_transient]
        obs_transient['mjd'] -= min(obs_transient['mjd'])  # so all transients start at time 0
        adf = np.vstack((adf, obs_transient))

    obs_transient = pd.DataFrame(data=adf, columns=obs_transient.columns)

    return obs_transient


def __generate_gp_all_objects(object_list, obs_transient):
    adf = pd.DataFrame(
        data=[],
        columns=["mjd", "lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz", "object_id"],
    )

    filters = obs_transient['filter']
    filters = list(np.unique(filters))
    gp_wavelengths = np.vectorize(pb_wavelengths.get)(filters)
    inverse_pb_wavelengths = {v: k for k, v in pb_wavelengths.items()}

    for object_id in object_list:

        df = obs_transient[obs_transient["object_id"] == object_id]

        gp_predict = fit_2d_gp(df)

        number_gp = 100
        gp_times = np.linspace(min(df["mjd"]), max(df["mjd"]), number_gp)
        obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
        obj_gps["filter"] = obj_gps["filter"].map(inverse_pb_wavelengths)

        obj_gps = obj_gps.pivot(index="mjd", columns="filter", values="flux")
        obj_gps = obj_gps.reset_index()
        obj_gps["object_id"] = object_id
        adf = np.vstack((adf, obj_gps))
    return pd.DataFrame(data=adf, columns=obj_gps.columns)


def __load_plasticc_dataset_from_csv():

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    data = pd.read_csv(
        f"{asnwd}/data/plasticc/training_set.csv",
        sep=",",
    )
    data = __remap_filters(df=data)
    data.rename(
        {"flux_err": "flux_error"}, axis="columns", inplace=True
    )  # snmachine and PLAsTiCC uses a different denomination


    df = __filter_dataframe_only_supernova(
        f"{asnwd}/data/plasticc/train_subset.txt",
        data,
    )

    object_list = list(np.unique(df['object_id']))

    obs_transient = __transient_trim(object_list, df)
    generated_gp_dataset = __generate_gp_all_objects(object_list, obs_transient)
    generated_gp_dataset['object_id'] = generated_gp_dataset['object_id'].astype(int)

    metadata_pd = pd.read_csv(
        f"{asnwd}/data/plasticc/training_set_metadata.csv",
        sep=",",
        index_col="object_id",
    )

    metadata_pd = metadata_pd.reset_index()
    metadata_pd['object_id'] = metadata_pd['object_id'].astype(int)

    df_with_labels = generated_gp_dataset.merge(metadata_pd, on='object_id', how='left')

    df = df_with_labels.drop(
        columns=[
            "ra",
            "decl",
            "gal_l",
            "gal_b",
            "ddf",
            "hostgal_specz",
            "hostgal_photoz",
            "hostgal_photoz_err",
            "distmod",
            "mwebv",
        ]
    )

    df.to_parquet(
        f"{asnwd}/data/plasticc/transformed_df.parquet",
        engine="pyarrow",
        compression="snappy",
    )

    return df


def load_plasticc(timesteps=20, step=20):

    try:
        df = pd.read_parquet(
            f"{asnwd}/data/plasticc/transformed_df.parquet",
            engine="pyarrow",
        )
    except IOError:
        df = __load_plasticc_dataset_from_csv()

    cols = ['lsstg', 'lssti', 'lsstr', 'lsstu', 'lssty', 'lsstz']

    df_train, df_val, df_test, num_features = train_val_test_split(df, cols)
    assert num_features == 6

    robust_scale(df_train, df_val, df_test, cols)

    TIME_STEPS = timesteps
    STEP = step

    X_train, y_train = create_dataset(
        df_train[cols],
        df_train.target,
        TIME_STEPS,
        STEP
    )

    X_val, y_val = create_dataset(
        df_val[cols],
        df_val.target,
        TIME_STEPS,
        STEP
    )

    X_test, y_test = create_dataset(
        df_test[cols],
        df_test.target,
        TIME_STEPS,
        STEP
    )

    # Recalcuate for plasticc
    # assert y_train.shape == (95603, 1)

    return X_train, y_train, X_val, y_val, X_test, y_test
