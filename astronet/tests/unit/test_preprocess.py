import numpy as np
import os
import pandas as pd
import pytest

from pathlib import Path

from astronet.constants import pb_wavelengths, astronet_working_directory as asnwd
from astronet.preprocess import predict_2d_gp, fit_2d_gp, one_hot_encode
from astronet.utils import __transient_trim, __filter_dataframe_only_supernova, __remap_filters
from astronet.utils import load_wisdm_2010, load_wisdm_2019


def test_one_hot_encode():

    X_train, y_train, X_test, y_test = load_wisdm_2010()

    assert len(np.unique(y_train)) == 6

    # One hot encode y
    enc, y_train, y_test = one_hot_encode(y_train, y_test)

    assert y_train.shape[1] == 6
    assert y_test.shape[1] == 6

    del enc, X_train, y_train, X_test, y_test


@pytest.mark.skipif(os.getenv("CI") is not None, reason="Requires large 'phone.df' file")
def test_one_hot_encode_local():

    X_train, y_train, X_test, y_test = load_wisdm_2019()
    # One hot encode y
    enc, y_train, y_test = one_hot_encode(y_train, y_test)

    assert y_train.shape[1] == 18
    assert y_test.shape[1] == 18


def test_plasticc_fit_2d_gp():
    # See tests/func/test_gp_interpolation.py
    pass


@pytest.mark.skipif(os.getenv("CI") is not None, reason="Unable to find file on CI. Test locally.")
def test_plasticc_predict_2d_gp():

    data = pd.read_csv(
        f"{Path(__file__).absolute().parent.parent.parent.parent}/data/plasticc/training_set.csv",
        sep=",",
    )
    data = __remap_filters(df=data)
    data.rename(
        {"flux_err": "flux_error"}, axis="columns", inplace=True
    )  # snmachine and PLAsTiCC uses a different denomination

    filters = data['filter']
    filters = list(np.unique(filters))

    df = __filter_dataframe_only_supernova(
        f"{Path(__file__).absolute().parent.parent.parent.parent}/data/plasticc/train_subset.txt",
        data,
    )

    object_list = list(np.unique(df['object_id']))
    object_list = object_list[2:3]
    assert object_list == [1124]

    obs_transient_single = __transient_trim(object_list, df)
    gp_predict = fit_2d_gp(obs_transient_single)
    number_gp = 100
    gp_times = np.linspace(min(obs_transient_single['mjd']), max(obs_transient_single['mjd']), number_gp)
    gp_wavelengths = np.vectorize(pb_wavelengths.get)(filters)

    obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
    inverse_pb_wavelengths = {v: k for k, v in pb_wavelengths.items()}
    obj_gps['filter'] = obj_gps['filter'].map(inverse_pb_wavelengths)

    gp_head_values_truth = np.array(
        [
            [8.2662012, 5.26646831],
            [7.03501702, 4.61751857],
            [5.87325113, 3.97806474],
            [4.77776141, 3.35145995],
            [3.74378163, 2.74487713],
        ]
    )
    assert np.allclose(obj_gps[['flux', 'flux_error']].head().values, gp_head_values_truth)
