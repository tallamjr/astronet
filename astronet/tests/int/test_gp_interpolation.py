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

import numpy as np
import pandas as pd
import pytest

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.constants import LSST_FILTER_MAP, LSST_PB_WAVELENGTHS
from astronet.preprocess import fit_2d_gp, predict_2d_gp
from astronet.utils import (
    __filter_dataframe_only_supernova,
    __transient_trim,
    remap_filters,
)


@pytest.mark.skipif(
    os.getenv("CI") is not None, reason="Unable to find file on CI. Test locally."
)
def test_plasticc_gp_interpolation():

    data = pd.read_csv(
        f"{asnwd}/data/plasticc/training_set.csv",
        sep=",",
    )
    data = remap_filters(df=data, filter_map=LSST_FILTER_MAP)
    data.rename({"flux_err": "flux_error"}, axis="columns", inplace=True)

    filters = data["filter"]
    filters = list(np.unique(filters))
    assert len(filters) == 6

    assert data.shape == (1421705, 6)
    df = __filter_dataframe_only_supernova(
        f"{asnwd}/data/plasticc/train_subset.txt",
        data,
    )
    assert df.shape == (764572, 6)

    object_list = list(np.unique(df["object_id"]))
    assert len(object_list) == 3990
    object_list = object_list[2:3]
    assert object_list == [1124]

    obs_transient_single, _ = __transient_trim(object_list, df)

    gp_predict = fit_2d_gp(obs_transient_single)
    number_gp = 100
    gp_times = np.linspace(
        min(obs_transient_single["mjd"]), max(obs_transient_single["mjd"]), number_gp
    )
    assert gp_times.shape == (100,)

    gp_wavelengths = np.vectorize(LSST_PB_WAVELENGTHS.get)(filters)
    assert gp_wavelengths.shape == (6,)

    obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
    inverse_pb_wavelengths = {v: k for k, v in LSST_PB_WAVELENGTHS.items()}
    obj_gps["filter"] = obj_gps["filter"].map(inverse_pb_wavelengths)

    assert obj_gps["flux"].iat[0] == pytest.approx(8.266201199288844, 0.0000001)
    assert obj_gps["flux"].iat[3] == pytest.approx(4.777761407937418, 0.0000001)
