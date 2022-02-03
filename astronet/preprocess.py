import george
import pandas as pd
import numpy as np
import scipy.optimize as op
import tensorflow as tf

from astropy.table import Table, vstack
from functools import partial
from typing import List, Dict, Union

from astronet.constants import LSST_PB_WAVELENGTHS


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
    >>> fit_2d_gp
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
    >>> fit_2d_gp
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


def robust_scale(
    dataframe: pd.DataFrame, scale_columns: List[Union[str, int]]
) -> pd.DataFrame:
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
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    scaler = scaler.fit(dataframe[scale_columns])
    dataframe.loc[:, scale_columns] = scaler.transform(
        dataframe[scale_columns].to_numpy()
    )


def one_hot_encode(y_train, y_test):
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
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    enc = enc.fit(y_train)

    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)

    return enc, y_train, y_test


def tf_one_hot_encode(y_train, y_test):
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

    dct = {42: 0, 62: 1, 90: 2}

    lst = y_train.flatten().tolist()
    flabels = list(map(dct.get, lst))
    y_train = tf.one_hot(flabels, len(np.unique(y_train)))

    lst = y_test.flatten().tolist()
    flabels = list(map(dct.get, lst))
    y_test = tf.one_hot(flabels, len(np.unique(y_test)))

    return y_train, y_test
