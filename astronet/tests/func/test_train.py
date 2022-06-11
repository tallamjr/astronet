import inspect
import json
import random as python_random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.metrics import WeightedLogLoss
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

    @pytest.mark.parametrize(
        ("architecture", "dataset", "hyperrun", "wloss"),
        (
            (
                "atx",
                "plasticc",
                "scaledown-by-4",
                1.79,
            ),
            (
                "t2",
                "plasticc",
                "1613517996-0a72904",
                2.19,
            ),
            (
                "tinho",
                "plasticc",
                "1613517996-0a72904",
                1.99,
            ),
        ),
    )
    def test_train_UGRIZY_wZ(self, architecture, dataset, hyperrun, wloss):

        params = {
            "epochs": 2,
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
        # loss = training.get_wloss
        assert wloss == pytest.approx(loss, 0.01)
        tf.keras.backend.clear_session()

    @pytest.mark.xfail(reason="Pending results...")
    @pytest.mark.parametrize(
        ("architecture", "dataset", "hyperrun", "wloss"),
        (
            # ( TODO
            #     "atx",
            #     "plasticc",
            #     "scaledown-by-4",
            #     1.79,
            # ),
            (
                "t2",
                "plasticc",
                "1613517996-0a72904",
                2.19,
            ),
            # ( TODO
            #     "t2",
            #     "plasticc",
            #     "1613517996-0a72904",
            #     2.19,
            # ),
        ),
    )
    def test_train_UGRIZY_noZ(self, architecture, dataset, hyperrun, wloss):

        params = {
            "epochs": 2,
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
        # loss = training.get_wloss
        assert wloss == pytest.approx(loss, 0.01)
        tf.keras.backend.clear_session()

    @pytest.mark.xfail(reason="Pending results...")
    @pytest.mark.parametrize(
        ("architecture", "dataset", "hyperrun", "wloss"),
        (
            (
                "atx",
                "plasticc",
                "scaledown-by-4",
                1.79,
            ),
            (
                "t2",
                "plasticc",
                "1613517996-0a72904",
                2.19,
            ),
            (
                "tinho",
                "plasticc",
                "1613517996-0a72904",
                1.99,
            ),
        ),
    )
    def test_train_GR_noZ(self, architecture, dataset, hyperrun, wloss):

        params = {
            "epochs": 2,
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
        # loss = training.get_wloss
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
