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

from functools import partial
from typing import Dict, List, Union

import george
import numpy as np
import pandas as pd
import scipy.optimize as op
from astropy.table import Table, vstack

from astronet.constants import LSST_PB_WAVELENGTHS


def __filter_dataframe_only_supernova(
    object_list_filename: str, dataframe: pd.DataFrame
) -> pd.DataFrame:
    """Filter dataframe that contains many classes to only Supernovae types.

    Parameters
    ----------
    object_list_filename: str
        Path to txt file that contains the 'object_id' of the objects to _keep_ i.e. that are
        Supernovae.
    df: pd.DataFrame
        DataFrame containing all objects of various classes

    Returns
    -------
    filtered_dataframe: pd.DataFrame
        Transformed datafram containing only supernovae objects

    Examples
    --------
    >>> if snonly is not None:
    ...     dataform = "snonly"
    ...     df = __filter_dataframe_only_supernova(
    ...         f"{asnwd}/data/plasticc/train_subset.txt",
    ...         data,
    ...     )
    ... else:
    ...     dataform = "full"
    >>> df = data
    >>> object_list = list(np.unique(df["object_id"]))
    """
    plasticc_object_list = np.genfromtxt(object_list_filename, dtype="U")
    filtered_dataframe = dataframe[dataframe["object_id"].isin(plasticc_object_list)]
    return filtered_dataframe


def __transient_trim(
    object_list: List[str], df: pd.DataFrame
) -> (pd.DataFrame, List[np.array]):
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
    adf = pd.DataFrame(data=[], columns=df.columns)
    good_object_list = []
    for obj in object_list:
        obs = df[df["object_id"] == obj]
        obs_time = obs["mjd"]
        obs_detected_time = obs_time[obs["detected"] == 1]
        if len(obs_detected_time) == 0:
            print(f"Zero detected points for object:{object_list.index(obj)}")
            continue
        is_obs_transient = (obs_time > obs_detected_time.iat[0] - 50) & (
            obs_time < obs_detected_time.iat[-1] + 50
        )
        obs_transient = obs[is_obs_transient]
        if len(obs_transient["mjd"]) == 0:
            is_obs_transient = (obs_time > obs_detected_time.iat[0] - 1000) & (
                obs_time < obs_detected_time.iat[-1] + 1000
            )
            obs_transient = obs[is_obs_transient]
        obs_transient["mjd"] -= min(
            obs_transient["mjd"]
        )  # so all transients start at time 0
        good_object_list.append(object_list.index(obj))
        adf = np.vstack((adf, obs_transient))

    obs_transient = pd.DataFrame(data=adf, columns=obs_transient.columns)

    filter_indices = good_object_list
    axis = 0
    array = np.array(object_list)

    new_filtered_object_list = np.take(array, filter_indices, axis)

    return obs_transient, list(new_filtered_object_list)


def fit_2d_gp(
    obj_data: pd.DataFrame,
    return_kernel: bool = False,
    pb_wavelengths: Dict = LSST_PB_WAVELENGTHS,
    **kwargs,
):
    """Fit a 2D Gaussian process.

    If required, predict the GP at evenly spaced points along a light curve.

    Parameters
    ----------
    obj_data : pd.DataFrame
        Time, flux and flux error of the data (specific filter of an object).
    return_kernel : bool, default = False
        Whether to return the used kernel.
    pb_wavelengths: dict
        Mapping of the passband wavelengths for each filter used.
    kwargs : dict
        Additional keyword arguments that are ignored at the moment. We allow
        additional keyword arguments so that the various functions that
        call this one can be called with the same arguments.

    Returns
    -------
    kernel: george.gp.GP.kernel, optional
        The kernel used to fit the GP.
    gp_predict : functools.partial of george.gp.GP
        The GP instance that was used to fit the object.

    Examples
    --------
    >>> gp_wavelengths = np.vectorize(pb_wavelengths.get)(filters)
    >>> inverse_pb_wavelengths = {v: k for k, v in pb_wavelengths.items()}
    >>> gp_predict = fit_2d_gp(df, pb_wavelengths=pb_wavelengths)
    ...
    """
    guess_length_scale = 20.0  # a parameter of the Matern32Kernel

    obj_times = obj_data.mjd.astype(float)
    obj_flux = obj_data.flux.astype(float)
    obj_flux_error = obj_data.flux_error.astype(float)
    obj_wavelengths = obj_data["filter"].map(pb_wavelengths)

    def neg_log_like(p):  # Objective function: negative log-likelihood
        gp.set_parameter_vector(p)
        loglike = gp.log_likelihood(obj_flux, quiet=True)
        return -loglike if np.isfinite(loglike) else 1e25

    def grad_neg_log_like(p):  # Gradient of the objective function.
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(obj_flux, quiet=True)

    # Use the highest signal-to-noise observation to estimate the scale. We
    # include an error floor so that in the case of very high
    # signal-to-noise observations we pick the maximum flux value.
    signal_to_noises = np.abs(obj_flux) / np.sqrt(
        obj_flux_error**2 + (1e-2 * np.max(obj_flux)) ** 2
    )
    scale = np.abs(obj_flux[signal_to_noises.idxmax()])

    kernel = (0.5 * scale) ** 2 * george.kernels.Matern32Kernel(
        [guess_length_scale**2, 6000**2], ndim=2
    )
    kernel.freeze_parameter("k2:metric:log_M_1_1")

    gp = george.GP(kernel)
    default_gp_param = gp.get_parameter_vector()
    x_data = np.vstack([obj_times, obj_wavelengths]).T
    gp.compute(x_data, obj_flux_error)

    bounds = [(0, np.log(1000**2))]
    bounds = [(default_gp_param[0] - 10, default_gp_param[0] + 10)] + bounds
    results = op.minimize(
        neg_log_like,
        gp.get_parameter_vector(),
        jac=grad_neg_log_like,
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-6,
    )

    if results.success:
        gp.set_parameter_vector(results.x)
    else:
        # Fit failed. Print out a warning, and use the initial guesses for fit
        # parameters.
        obj = obj_data["object_id"][0]
        print("GP fit failed for {}! Using guessed GP parameters.".format(obj))
        gp.set_parameter_vector(default_gp_param)

    gp_predict = partial(gp.predict, obj_flux)

    if return_kernel:
        return kernel, gp_predict
    return gp_predict


def generate_gp_single_event(
    df: pd.DataFrame, timesteps: int = 100, pb_wavelengths: Dict = LSST_PB_WAVELENGTHS
) -> pd.DataFrame:
    """Intermediate helper function useful for visualisation of the original data with the mean of
    the Gaussian Process interpolation as well as the uncertainity.

    Additional steps required to build full dataframe for classification found in
    `generate_gp_all_objects`, namely:

        ...
        obj_gps = pd.pivot_table(obj_gps, index="mjd", columns="filter", values="flux")
        obj_gps = obj_gps.reset_index()
        obj_gps["object_id"] = object_id
        ...

    To allow a transformation from:

        mjd	        flux	    flux_error	filter
    0	0.000000	19.109279	0.176179	ztfg
    1	0.282785	19.111843	0.173419	ztfg
    2	0.565571	19.114406	0.170670	ztfg

    to ...

    filter	mjd	        ztfg    ztfr	object_id
    0	    0	        19.1093	19.2713	27955532126447639664866058596
    1	    0.282785	19.1118	19.2723	27955532126447639664866058596
    2	    0.565571	19.1144	19.2733	27955532126447639664866058596

    Examples
    --------
    >>> _obj_gps = generate_gp_single_event(data)
    >>> ax = plot_event_data_with_model(data, obj_model=_obj_gps, pb_colors=ZTF_PB_COLORS)
    """

    filters = df["filter"]
    filters = list(np.unique(filters))

    gp_wavelengths = np.vectorize(pb_wavelengths.get)(filters)
    inverse_pb_wavelengths = {v: k for k, v in pb_wavelengths.items()}

    gp_predict = fit_2d_gp(df, pb_wavelengths=pb_wavelengths)

    number_gp = timesteps
    gp_times = np.linspace(min(df["mjd"]), max(df["mjd"]), number_gp)
    obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
    obj_gps["filter"] = obj_gps["filter"].map(inverse_pb_wavelengths)

    return obj_gps


def generate_gp_all_objects(
    object_list: List[str],
    obs_transient: pd.DataFrame,
    timesteps: int = 100,
    pb_wavelengths: Dict = LSST_PB_WAVELENGTHS,
) -> pd.DataFrame:
    """Generate Gaussian Process interpolation for all objects within 'object_list'. Upon
    completion, a dataframe is returned containing a value for each time step across each passband.

    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    obs_transient: pd.DataFrame
        Dataframe containing observational points with the transient section of the full light curve
    timesteps: int
        Number of points one would like to interpolate, i.e. how many points along the time axis
        should the Gaussian Process be evaluated
    pb_wavelengths: Dict
        A mapping of passbands and the associated wavelengths, specific to each survey. Current
        options are ZTF or LSST

    Returns
    -------
    df: pd.DataFrame(data=adf, columns=obj_gps.columns)
        Dataframe with the mean of the GP for N x timesteps

    Examples
    --------
    >>> object_list = list(np.unique(df["object_id"]))
    >>> obs_transient, object_list = __transient_trim(object_list, df)
    >>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    filters = obs_transient["filter"]
    filters = list(np.unique(filters))

    columns = []
    columns.append("mjd")
    for filt in filters:
        columns.append(filt)
    columns.append("object_id")

    adf = pd.DataFrame(
        data=[],
        columns=columns,
    )

    for object_id in object_list:
        print(f"OBJECT ID:{object_id} at INDEX:{object_list.index(object_id)}")
        df = obs_transient[obs_transient["object_id"] == object_id]

        obj_gps = generate_gp_single_event(df, timesteps, pb_wavelengths)

        obj_gps = pd.pivot_table(obj_gps, index="mjd", columns="filter", values="flux")
        obj_gps = obj_gps.reset_index()
        obj_gps["object_id"] = object_id
        adf = np.vstack((adf, obj_gps))
    return pd.DataFrame(data=adf, columns=obj_gps.columns)


def predict_2d_gp(gp_predict, gp_times, gp_wavelengths):
    """Outputs the predictions of a Gaussian Process.

    Parameters
    ----------
    gp_predict : functools.partial of george.gp.GP
        The GP instance that was used to fit the object.
    gp_times : numpy.ndarray
        Times to evaluate the Gaussian Process at.
    gp_wavelengths : numpy.ndarray
        Wavelengths to evaluate the Gaussian Process at.

    Returns
    -------
    obj_gps : pandas.core.frame.DataFrame, optional
        Time, flux and flux error of the fitted Gaussian Process.

    Examples
    --------
    >>> gp_predict = fit_2d_gp(df, pb_wavelengths=pb_wavelengths)
    >>> number_gp = timesteps
    >>> gp_times = np.linspace(min(df["mjd"]), max(df["mjd"]), number_gp)
    >>> obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
    >>> obj_gps["filter"] = obj_gps["filter"].map(inverse_pb_wavelengths)
    ...
    """
    unique_wavelengths = np.unique(gp_wavelengths)
    number_gp = len(gp_times)
    obj_gps = []
    for wavelength in unique_wavelengths:
        gp_wavelengths = np.ones(number_gp) * wavelength
        pred_x_data = np.vstack([gp_times, gp_wavelengths]).T
        pb_pred, pb_pred_var = gp_predict(pred_x_data, return_var=True)
        # stack the GP results in a array momentarily
        obj_gp_pb_array = np.column_stack((gp_times, pb_pred, np.sqrt(pb_pred_var)))
        obj_gp_pb = Table(
            [
                obj_gp_pb_array[:, 0],
                obj_gp_pb_array[:, 1],
                obj_gp_pb_array[:, 2],
                [wavelength] * number_gp,
            ],
            names=["mjd", "flux", "flux_error", "filter"],
        )
        if len(obj_gps) == 0:  # initialize the table for 1st passband
            obj_gps = obj_gp_pb
        else:  # add more entries to the table
            obj_gps = vstack((obj_gps, obj_gp_pb))

    obj_gps = obj_gps.to_pandas()
    return obj_gps


def remap_filters(df: pd.DataFrame, filter_map: Dict) -> pd.DataFrame:
    """Function to remap integer filters to the corresponding filters.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe of lightcurve observations
    filter_map: dict
        Corresponding map for filters used. Current options are found in astronet.constants:
        {LSST_FILTER_MAP, ZTF_FILTER_MAP}
    """
    df.rename({"passband": "filter"}, axis="columns", inplace=True)
    df["filter"].replace(to_replace=filter_map, inplace=True)
    return df


def robust_scale(
    dataframe: pd.DataFrame, scale_columns: List[Union[str, int]]
) -> pd.DataFrame:
    """Standardize a dataset along axis=0 (rows)

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing GP interolated values
    scale_columns: List[Union[str, int]]
        Which coloums to keep when applying the transformation

    Returns
    -------
    <Inpace Operation>

    Examples
    --------
    >>> df = __load_plasticc_dataset_from_csv(timesteps)
    >>> cols = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
    >> robust_scale(df, cols)
    """
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    scaler = scaler.fit(dataframe[scale_columns])
    dataframe.loc[:, scale_columns] = scaler.transform(
        dataframe[scale_columns].to_numpy()
    )
    return dataframe


def one_hot_encode(y_train, y_test):
    """Encode categorical features as a one-hot numeric array.

    Examples
    --------
    >>> # Load data
    >>> X_train, y_train, X_test, y_test = load_wisdm_2010()
    >>> # One hot encode y
    >>> enc, y_train, y_test = one_hot_encode(y_train, y_test)
    >>> encoding_file = f"{asnwd}/data/{dataset}.encoding"
    >>> if not os.path.exists(encoding_file):
    ...     with open(encoding_file, "wb") as f:
    ...         joblib.dump(enc, f)
    """
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    enc = enc.fit(y_train)

    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)

    return enc, y_train, y_test
