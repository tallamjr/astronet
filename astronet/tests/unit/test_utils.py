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

import pytest

from astronet.metrics import WeightedLogLoss
from astronet.tests.conftest import SKIP_IF_M1
from astronet.utils import (
    load_dataset,
    load_plasticc,
    load_wisdm_2010,
    load_wisdm_2019,
)


@pytest.mark.skipif(os.getenv("CI") is not None, reason="To be checked locally")
@SKIP_IF_M1
def test_load_wisdm_2010():

    X_train, y_train, X_test, y_test = load_wisdm_2010(timesteps=200, step=200)

    assert X_train.shape == (4118, 200, 3)
    assert X_test.shape == (1373, 200, 3)

    assert y_train.shape == (4118, 1)
    assert y_test.shape == (1373, 1)


@pytest.mark.skipif(
    os.getenv("CI") is not None, reason="Requires large 'phone.df' file"
)
def test_load_wisdm_2019():

    X_train, y_train, X_test, y_test = load_wisdm_2019(timesteps=100, step=40)

    assert X_train.shape == (89628, 100, 3)
    assert X_test.shape == (29876, 100, 3)

    assert y_train.shape == (89628, 1)
    assert y_test.shape == (29876, 1)


@pytest.mark.skipif(os.getenv("CI") is not None, reason="Requires large datafile")
def test_load_plasticc_snonly():

    X_train, y_train, X_test, y_test = load_plasticc(
        timesteps=100, step=100, snonly=True
    )

    assert X_train.shape == (2991, 100, 6)
    assert X_test.shape == (998, 100, 6)

    assert y_train.shape == (2991, 1)
    assert y_test.shape == (998, 1)


@pytest.mark.skipif(os.getenv("CI") is not None, reason="Requires large datafile")
def test_load_plasticc():

    X_train, y_train, X_test, y_test = load_plasticc(timesteps=100, step=100)

    assert X_train.shape == (5885, 100, 6)
    assert X_test.shape == (1962, 100, 6)

    assert y_train.shape == (5885, 1)
    assert y_test.shape == (1962, 1)


@pytest.mark.skipif(os.getenv("CI") is not None, reason="Requires large datafile")
def test_load_dataset():

    X_train, y_train, X_test, y_test, loss = load_dataset("plasticc")

    assert X_train.shape == (5885, 100, 6)
    assert X_test.shape == (1962, 100, 6)

    assert y_train.shape == (5885, 14)
    assert y_test.shape == (1962, 14)

    assert type(WeightedLogLoss()) == type(loss)


@pytest.mark.skipif(os.getenv("CI") is not None, reason="Requires large datafile")
@pytest.mark.xfail(
    reason="Hardcoded dataform set to 'testset' for now: see commit: fd02ac"
)
def test_load_dataset_with_z():

    dataset = "plasticc"
    X_train, y_train, X_test, y_test, loss, ZX_train, ZX_test = load_dataset(
        dataset, redshift=True, snonly=True
    )

    assert X_train.shape == (2991, 100, 6)
    assert X_test.shape == (998, 100, 6)

    assert y_train.shape == (2991, 3)
    assert y_test.shape == (998, 3)

    assert ZX_train.shape == (2991, 2)
    assert ZX_test.shape == (998, 2)

    Z_input_shape_nobatch = (2,)
    assert Z_input_shape_nobatch == ZX_train.shape[1:]

    assert type(WeightedLogLoss()) == type(loss)
