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

import inspect
import random as python_random

import numpy as np
import pytest
import tensorflow as tf

from astronet.tests.conftest import ISA
from astronet.train import Training
from astronet.utils import astronet_logger

log = astronet_logger(__file__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(RANDOM_SEED)


@pytest.mark.skipif(ISA != "arm64", reason="Only run this particular test locally")
class TestTrain:
    """A class with common parameters, `architecture`, `dataset` and the `hyperrun`."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.epochs = 2

    @pytest.mark.parametrize(
        ("architecture", "dataset", "hyperrun", "wloss"),
        (
            (
                "atx",
                "plasticc",
                "scaledown-by-4",
                4.35,
            ),
            (
                "t2",
                "plasticc",
                "1613517996-0a72904",
                2.03,
            ),
            (
                "tinho",
                "plasticc",
                "1613517996-0a72904",
                1.97,
            ),
        ),
    )
    def test_train_UGRIZY_wZ(self, architecture, dataset, hyperrun, wloss):

        params = {
            "epochs": self.epochs,
            "architecture": architecture,
            "dataset": dataset,
            "model": hyperrun,
            "testset": True,
            "redshift": True,
            "fink": None,
            "avocado": None,
        }
        log.info(f"\n{params}")

        training = Training(**params)

        loss = training()
        assert wloss == pytest.approx(loss, 0.01)
        tf.keras.backend.clear_session()

    # @pytest.mark.xfail(reason="Pending results...")
    @pytest.mark.parametrize(
        ("architecture", "dataset", "hyperrun", "wloss"),
        (
            (
                "atx",
                "plasticc",
                "scaledown-by-4",
                5.03,
            ),
            (
                "t2",
                "plasticc",
                "1613517996-0a72904",
                2.10,
            ),
            (
                "tinho",
                "plasticc",
                "1613517996-0a72904",
                2.03,
            ),
        ),
    )
    def test_train_UGRIZY_noZ(self, architecture, dataset, hyperrun, wloss):

        params = {
            "epochs": self.epochs,
            "architecture": architecture,
            "dataset": dataset,
            "model": hyperrun,
            "testset": True,
            "redshift": None,
            "fink": None,
            "avocado": None,
        }
        log.info(f"\n{params}")

        training = Training(**params)

        loss = training()
        assert wloss == pytest.approx(loss, 0.01)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        ("architecture", "dataset", "hyperrun", "wloss"),
        (
            (
                "atx",
                "plasticc",
                "scaledown-by-4",
                4.94,
            ),
            (
                "t2",
                "plasticc",
                "1613517996-0a72904",
                2.03,
            ),
            (
                "tinho",
                "plasticc",
                "1613517996-0a72904",
                2.08,
            ),
        ),
    )
    def test_train_GR_noZ(self, architecture, dataset, hyperrun, wloss):

        params = {
            "epochs": self.epochs,
            "architecture": architecture,
            "dataset": dataset,
            "model": hyperrun,
            "testset": True,
            "redshift": None,
            "fink": True,
            "avocado": None,
        }
        log.info(f"\n{params}")

        training = Training(**params)

        loss = training()
        assert wloss == pytest.approx(loss, 0.01)
        tf.keras.backend.clear_session()

    @pytest.mark.xfail(reason="Pending results...")
    # TODO: ALL
    # @pytest.mark.parametrize(
    #     ("architecture", "dataset", "hyperrun", "wloss"),
    #     (
    # (
    #     "atx",
    #     "plasticc",
    #     "scaledown-by-4",
    #     1.79,
    # ),
    # (
    #     "t2",
    #     "plasticc",
    #     "1613517996-0a72904",
    #     2.19,
    # ),
    # (
    #     "tinho",
    #     "plasticc",
    #     "1613517996-0a72904",
    #     1.99,
    # ),
    # ),
    # )
    def test_train_GR_wZ(self, architecture, dataset, hyperrun, wloss):
        log.warning(f"{inspect.stack()[0].function} -- Not Implemented Yet")
        pass
