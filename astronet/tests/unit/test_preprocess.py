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

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.constants import LSST_FILTER_MAP, LSST_PB_WAVELENGTHS
from astronet.preprocess import (
    fit_2d_gp,
    one_hot_encode,
    predict_2d_gp,
)
from astronet.tests.conftest import SKIP_IF_M1
from astronet.utils import (
    __filter_dataframe_only_supernova,
    __transient_trim,
    load_wisdm_2010,
    load_wisdm_2019,
    remap_filters,
)


@pytest.mark.skipif(os.getenv("CI") is not None, reason="To be checked locally")
@SKIP_IF_M1
def test_one_hot_encode():

    X_train, y_train, X_test, y_test = load_wisdm_2010()

    assert len(np.unique(y_train)) == 6

    # One hot encode y
    enc, y_train, y_test = one_hot_encode(y_train, y_test)

    assert y_train.shape[1] == 6
    assert y_test.shape[1] == 6

    del enc, X_train, y_train, X_test, y_test


@pytest.mark.skipif(
    os.getenv("CI") is not None, reason="Requires large 'phone.df' file"
)
def test_one_hot_encode_local():

    X_train, y_train, X_test, y_test = load_wisdm_2019()
    # One hot encode y
    enc, y_train, y_test = one_hot_encode(y_train, y_test)

    assert y_train.shape[1] == 18
    assert y_test.shape[1] == 18


def test_plasticc_fit_2d_gp():
    # See tests/func/test_gp_interpolation.py
    pass


@pytest.mark.skipif(
    os.getenv("CI") is not None, reason="Unable to find file on CI. Test locally."
)
def test_plasticc_predict_2d_gp():

    data = pd.read_csv(
        f"{Path(__file__).absolute().parent.parent.parent.parent}/data/plasticc/training_set.csv",
        sep=",",
    )
    data = remap_filters(df=data, filter_map=LSST_FILTER_MAP)
    data.rename(
        {"flux_err": "flux_error"}, axis="columns", inplace=True
    )  # snmachine and PLAsTiCC uses a different denomination

    filters = data["filter"]
    filters = list(np.unique(filters))

    df = __filter_dataframe_only_supernova(
        f"{Path(__file__).absolute().parent.parent.parent.parent}/data/plasticc/train_subset.txt",
        data,
    )

    object_list = list(np.unique(df["object_id"]))
    object_list = object_list[2:3]
    assert object_list == [1124]

    obs_transient_single, _ = __transient_trim(object_list, df)
    gp_predict = fit_2d_gp(obs_transient_single)
    number_gp = 100
    gp_times = np.linspace(
        min(obs_transient_single["mjd"]), max(obs_transient_single["mjd"]), number_gp
    )
    gp_wavelengths = np.vectorize(LSST_PB_WAVELENGTHS.get)(filters)

    obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
    inverse_pb_wavelengths = {v: k for k, v in LSST_PB_WAVELENGTHS.items()}
    obj_gps["filter"] = obj_gps["filter"].map(inverse_pb_wavelengths)

    gp_head_values_truth = np.array(
        [
            [8.2662012, 5.26646831],
            [7.03501702, 4.61751857],
            [5.87325113, 3.97806474],
            [4.77776141, 3.35145995],
            [3.74378163, 2.74487713],
        ]
    )
    assert np.allclose(
        obj_gps[["flux", "flux_error"]].head().values, gp_head_values_truth
    )
