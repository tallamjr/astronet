import joblib
import logging
import os.path
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from pathlib import Path
from scipy import stats
from sklearn import model_selection

from astronet.constants import (
    pb_wavelengths,
    astronet_working_directory as asnwd,
)
from astronet.metrics import WeightedLogLoss
from astronet.preprocess import (
    robust_scale,
    fit_2d_gp,
    predict_2d_gp,
    one_hot_encode,
)


# 'SettingWithCopyWarning' in Pandas: https://bit.ly/3mv3fhw
pd.options.mode.chained_assignment = None  # default='warn'


def astronet_logger(name, level="INFO"):
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


def find_optimal_batch_size(training_set_length):

    if (training_set_length < 10000):
        batch_size_list = [16, 32, 64]
    else:
        batch_size_list = [512, 1024, 2028, 4096]
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

    features = df[cols]
    # column_indices = {name: i for i, name in enumerate(features.columns)}

    n = len(df)
    df_train = df[0 : int(n * 0.8)].copy()
    df_val = df[int(n * 0.8) : int(n * 0.95)].copy()
    df_test = df[int(n * 0.95) :].copy()

    num_features = features.shape[1]

    return df_train, df_val, df_test, num_features


def create_dataset(X, y, time_steps=1, step=1):

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
        str(Path(__file__).absolute().parent.parent) +
        "/data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt",
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

    Xs, ys = create_dataset(
        df[cols],
        df.activity,
        TIME_STEPS,
        STEP
    )

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        Xs, ys, random_state=RANDOM_SEED
    )

    return X_train, y_train, X_test, y_test


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

    Xs, ys = create_dataset(
        df[cols],
        df.activity,
        TIME_STEPS,
        STEP
    )

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        Xs, ys, random_state=RANDOM_SEED
    )

    return X_train, y_train, X_test, y_test


def load_mts(dataset):

    X_train = np.load(
        f"{asnwd}/data/transformed-mtsdata/{dataset}/x_train.npy"
    )
    y_train = np.load(
        f"{asnwd}/data/transformed-mtsdata/{dataset}/y_train.npy"
    )
    X_test = np.load(
        f"{asnwd}/data/transformed-mtsdata/{dataset}/x_test.npy"
    )
    y_test = np.load(
        f"{asnwd}/data/transformed-mtsdata/{dataset}/y_test.npy"
    )

    return X_train, y_train, X_test, y_test


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
    good_object_list = []
    for obj in object_list:
        obs = df[df['object_id'] == obj]
        obs_time = obs['mjd']
        obs_detected_time = obs_time[obs['detected'] == 1]
        if len(obs_detected_time) == 0:
            print(f"Zero detected points for object:{object_list.index(obj)}")
            continue
        is_obs_transient = (obs_time > obs_detected_time.iat[0] - 50) & (obs_time < obs_detected_time.iat[-1] + 50)
        obs_transient = obs[is_obs_transient]
        if len(obs_transient['mjd']) == 0:
            is_obs_transient = (obs_time > obs_detected_time.iat[0] - 1000) & (obs_time < obs_detected_time.iat[-1] + 1000)
            obs_transient = obs[is_obs_transient]
        obs_transient['mjd'] -= min(obs_transient['mjd'])  # so all transients start at time 0
        good_object_list.append(object_list.index(obj))
        adf = np.vstack((adf, obs_transient))

    obs_transient = pd.DataFrame(data=adf, columns=obs_transient.columns)

    filter_indices = good_object_list
    axis = 0
    array = np.array(object_list)

    new_filtered_object_list = np.take(array, filter_indices, axis)

    return obs_transient, list(new_filtered_object_list)


def __generate_gp_all_objects(object_list, obs_transient, timesteps):
    adf = pd.DataFrame(
        data=[],
        columns=["mjd", "lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz", "object_id"],
    )

    # import pdb;pdb.set_trace()
    filters = obs_transient['filter']
    filters = list(np.unique(filters))
    gp_wavelengths = np.vectorize(pb_wavelengths.get)(filters)
    inverse_pb_wavelengths = {v: k for k, v in pb_wavelengths.items()}

    for object_id in object_list:
        print(f"OBJECT ID:{object_id} at INDEX:{object_list.index(object_id)}")
        df = obs_transient[obs_transient["object_id"] == object_id]

        gp_predict = fit_2d_gp(df)

        number_gp = timesteps
        gp_times = np.linspace(min(df["mjd"]), max(df["mjd"]), number_gp)
        obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
        obj_gps["filter"] = obj_gps["filter"].map(inverse_pb_wavelengths)

        obj_gps = pd.pivot_table(obj_gps, index="mjd", columns="filter", values="flux")
        obj_gps = obj_gps.reset_index()
        obj_gps["object_id"] = object_id
        adf = np.vstack((adf, obj_gps))
    return pd.DataFrame(data=adf, columns=obj_gps.columns)


def __load_plasticc_dataset_from_csv(timesteps, snonly=None):

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

    if snonly is not None:
        dataform = "snonly"
        df = __filter_dataframe_only_supernova(
            f"{asnwd}/data/plasticc/train_subset.txt",
            data,
        )
    else:
        dataform = "full"
        df = data

    object_list = list(np.unique(df['object_id']))

    obs_transient = __transient_trim(object_list, df)
    generated_gp_dataset = __generate_gp_all_objects(object_list, obs_transient, timesteps)
    generated_gp_dataset['object_id'] = generated_gp_dataset['object_id'].astype(int)

    metadata_pd = pd.read_csv(
        f"{asnwd}/data/plasticc/training_set_metadata.csv",
        sep=",",
        index_col="object_id",
    )

    metadata_pd = metadata_pd.reset_index()
    metadata_pd['object_id'] = metadata_pd['object_id'].astype(int)

    df_with_labels = generated_gp_dataset.merge(metadata_pd, on='object_id', how='left')

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
    df['object_id'] = df['object_id'].astype(int)
    print(df.dtypes)

    print(df.columns)
    print(df.head())
    print(df.dtypes)

    df.to_csv(f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_with_z.csv")

    # df.to_parquet(
    #     f"{asnwd}/data/plasticc/{dataform}_transformed_df_timesteps_{timesteps}_with_z.parquet",
    #     engine="pyarrow",
    #     compression="snappy",
    # )

    return df


def __load_plasticc_test_set_dataset_from_csv(timesteps, snonly=None, batch_filename=None):

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # batch_filename = "plasticc_test_lightcurves_01"

    data = pd.read_csv(
        f"{asnwd}/data/plasticc/test_set/{batch_filename}.csv",
        sep=",",
    )
    data = __remap_filters(df=data)
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

    object_list = list(np.unique(df['object_id']))

    obs_transient = __transient_trim(object_list, df)
    generated_gp_dataset = __generate_gp_all_objects(object_list, obs_transient, timesteps)
    generated_gp_dataset['object_id'] = generated_gp_dataset['object_id'].astype(int)

    metadata_pd = pd.read_csv(
        f"{asnwd}/data/plasticc/test_set/plasticc_test_metadata.csv",
        sep=",",
        index_col="object_id",
    )

    metadata_pd = metadata_pd.reset_index()
    metadata_pd['object_id'] = metadata_pd['object_id'].astype(int)

    df_with_labels = generated_gp_dataset.merge(metadata_pd, on='object_id', how='left')

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
    df['object_id'] = df['object_id'].astype(int)
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


def __load_avocado_plasticc_dataset_from_csv(timesteps, snonly=None, batch_filename=None):

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    data = pd.read_csv(
        f"{asnwd}/data/plasticc/avocado/{batch_filename}.csv",
        sep=",",
    )

    data.rename(
        {"band": "filter"}, axis="columns", inplace=True
    )
    data.rename(
        {"time": "mjd"}, axis="columns", inplace=True
    )

    if snonly is not None:
        dataform = "snonly"
        df = __filter_dataframe_only_supernova(
            f"{asnwd}/data/plasticc/train_subset.txt",
            data,
        )
    else:
        dataform = "avocado"
        df = data

    object_list = list(np.unique(df['object_id']))

    obs_transient, object_list = __transient_trim(object_list, df)
    generated_gp_dataset = __generate_gp_all_objects(object_list, obs_transient, timesteps)
    # generated_gp_dataset['object_id'] = generated_gp_dataset['object_id'].astype(int)

    metadata_pd = pd.read_csv(
        f"{asnwd}/data/plasticc/avocado/avo_aug_0.csv",
        sep=",",
        index_col="object_id",
    )

    metadata_pd = metadata_pd.reset_index()
    # metadata_pd['object_id'] = metadata_pd['object_id'].astype(int)

    df_with_labels = generated_gp_dataset.merge(metadata_pd, on='object_id', how='left')

    df_with_labels.rename(
        {"class": "target"}, axis="columns", inplace=True
    )
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

    df.to_csv(f"{asnwd}/data/plasticc/avocado/{dataform}_transformed_df_timesteps_{timesteps}_with_z_{batch_filename}.csv")

    return df


def __generate_augmented_plasticc_dataset_from_pickle(augmented_binary):

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
    adf = adf.replace({'_aug': '000'}, regex=True)
    adf = adf.convert_dtypes()
    print(adf.dtypes)
    print(adf.head())
    print(adf['object_id'].values)
    np.savetxt(f"{asnwd}/data/plasticc/aug_object_list.txt", adf['object_id'].values, fmt='%s')

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

    data = data.replace({'_aug': '000'}, regex=True)
    data = data.convert_dtypes()

    # data = __remap_filters(df=data)
    # data.rename(
    #     {"flux_err": "flux_error"}, axis="columns", inplace=True
    # )  # snmachine and PLAsTiCC uses a different denomination

    print(data.head())

    # import pdb;pdb.set_trace()
    df = __filter_dataframe_only_supernova(
        f"{asnwd}/data/plasticc/aug_object_list.txt",
        data,
    )

    print(df.head())
    print(df.dtypes)

    object_list = list(np.unique(df['object_id']))
    print(len(object_list))

    # obs_transient = __transient_trim(object_list, df)
    generated_gp_dataset = __generate_gp_all_objects(object_list, df, timesteps)
    generated_gp_dataset['object_id'] = generated_gp_dataset['object_id'].astype(int)

    metadata_pd = pd.read_csv(
        f"{asnwd}/data/plasticc/augmented_training_set_metadata.csv",
        sep=",",
        index_col="object_id",
    )

    metadata_pd = metadata_pd.reset_index()
    metadata_pd = metadata_pd.replace({'_aug': '000'}, regex=True)
    metadata_pd = metadata_pd.convert_dtypes()
    metadata_pd['object_id'] = metadata_pd['object_id'].astype(int)

    df_with_labels = generated_gp_dataset.merge(metadata_pd, on='object_id', how='left')
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
    df['object_id'] = df['object_id'].astype(int)
    print(df.dtypes)

    print(df.columns)
    print(df.head())
    print(df.dtypes)

    # df.to_csv(f"{asnwd}/data/plasticc/augmented_transformed_df_timesteps_{timesteps}_with_z_backup.csv")
    df.to_csv(f"{asnwd}/data/plasticc/augmented_transformed_df_timesteps_{timesteps}_with_z.csv")

    # try:
    #     df.to_parquet(
    #         f"{asnwd}/data/plasticc/augmented_transformed_df_timesteps_{timesteps}_with_z.parquet",
    #         engine="pyarrow",
    #         compression="snappy",
    #     )

    # except IOError:
    #     df.to_csv(f"{asnwd}/data/plasticc/augmented_transformed_df_timesteps_{timesteps}_with_z.csv")

    return df


def load_plasticc(timesteps=100, step=100, redshift=None, augmented=None, snonly=None, avocado=None):

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

    cols = ['lsstg', 'lssti', 'lsstr', 'lsstu', 'lssty', 'lsstz']
    robust_scale(df, cols)

    Xs, ys = create_dataset(
        df[cols],
        df.target,
        TIME_STEPS,
        STEP
    )

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
        zcols = ['hostgal_photoz', 'hostgal_photoz_err']
        robust_scale(df, zcols)

        ZXs, zys = create_dataset(
            df[zcols],
            df.target,
            TIME_STEPS,
            STEP
        )

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


def load_full_avocado_plasticc_from_numpy(timesteps=100, redshift=None, batch_filename=None):

    dataform = "avocado"
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
        X_train, y_train, Z_train = save_avocado_training_set(batch_filename=batch_filename)

    if redshift is not None:
        return X_train, y_train, X_full_test_no_99, y_full_test_no_99, Z_train, Z_full_test_no_99
    else:
        return X_train, y_train


def load_full_plasticc_test_from_numpy(timesteps=100, redshift=None):

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    try:
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
            if (y_full_test.flatten()[i] in [991, 992, 993, 994]):
                pass
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
        df = __load_avocado_plasticc_dataset_from_csv(timesteps, batch_filename=batch_filename)


    cols = ['lsstg', 'lssti', 'lsstr', 'lsstu', 'lssty', 'lsstz']
    robust_scale(df, cols)

    Xs, ys = create_dataset(
        df[cols],
        df.target,
        TIME_STEPS,
        STEP
    )

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
        zcols = ['hostgal_photoz', 'hostgal_photoz_err']
        robust_scale(df, zcols)

        ZXs, zys = create_dataset(
            df[zcols],
            df.target,
            TIME_STEPS,
            STEP
        )

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


def save_plasticc_test_set(timesteps=100, step=100, redshift=None, augmented=None, snonly=None,
        batch_filename=None):

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
            df = __load_plasticc_test_set_dataset_from_csv(timesteps, batch_filename=batch_filename)

    cols = ['lsstg', 'lssti', 'lsstr', 'lsstu', 'lssty', 'lsstz']
    robust_scale(df, cols)

    Xs, ys = create_dataset(
        df[cols],
        df.true_target,
        TIME_STEPS,
        STEP
    )

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
        zcols = ['hostgal_photoz', 'hostgal_photoz_err']
        robust_scale(df, zcols)

        ZXs, zys = create_dataset(
            df[zcols],
            df.true_target,
            TIME_STEPS,
            STEP
        )

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


def load_dataset(dataset, redshift=None, balance=None, augmented=None, snonly=None, avocado=None, testset=None):
    if dataset == "wisdm_2010":
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
        # Load data
        X_train, y_train, X_test, y_test = load_mts(dataset)
        # transform the labels from integers to one hot vectors
        import sklearn
        enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
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
                X_train, y_train, X_test, y_test = load_full_avocado_plasticc_from_numpy(redshift=redshift)
            else:
                X_train, y_train, X_test, y_test = load_plasticc(augmented=augmented, snonly=snonly,
                        avocado=avocado)
        else:
            if testset is not None:
                X_train, y_train, X_test, y_test, ZX_train, ZX_test = load_full_plasticc_test_from_numpy(redshift=redshift)
            # if avocado is not None:
            #     X_train, y_train, X_test, y_test, ZX_train, ZX_test = load_full_avocado_plasticc_from_numpy(redshift=redshift)
            else:
                X_train, y_train, X_test, y_test, ZX_train, ZX_test = load_plasticc(
                    redshift=redshift, augmented=augmented, snonly=snonly, avocado=avocado
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
        num_samples, timesteps, num_features = X_train.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)

        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        # random_state: if None, the random number generator is the RandomState instance used by np.random.

        from imblearn.under_sampling import RandomUnderSampler
        X_resampled, y_resampled = RandomUnderSampler(sampling_strategy="not minority").fit_resample(
            X_train.reshape(X_train.shape[0], -1), y_train
        )
        # Re-shape 2D data back to 3D original shape, i.e (BATCH_SIZE, timesteps, num_features)
        X_resampled = np.reshape(X_resampled, (X_resampled.shape[0], timesteps, num_features))

        if redshift is not None:
            num_z_samples, num_z_features = ZX_train.shape
            Z_resampled, _ = RandomUnderSampler(sampling_strategy="not minority").fit_resample(
                ZX_train, y_train
            )
            Z_resampled = np.reshape(Z_resampled, (Z_resampled.shape[0], num_z_features))

            ZX_train = Z_resampled

        X_train = X_resampled
        y_train = y_resampled

    if redshift is None:
        return X_train, y_train, X_test, y_test, loss
    else:
        return X_train, y_train, X_test, y_test, loss, ZX_train, ZX_test
