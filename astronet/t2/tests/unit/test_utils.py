import os
import pytest

from astronet.metrics import WeightedLogLoss
from astronet.utils import load_wisdm_2010, load_wisdm_2019, load_plasticc, load_dataset


def test_load_wisdm_2010():

    X_train, y_train, X_test, y_test = load_wisdm_2010(timesteps=200, step=200)

    assert X_train.shape == (4118, 200, 3)
    assert X_test.shape == (1373, 200, 3)

    assert y_train.shape == (4118, 1)
    assert y_test.shape == (1373, 1)


@pytest.mark.skipif(os.getenv("CI") is not None, reason="Requires large 'phone.df' file")
def test_load_wisdm_2019():

    X_train, y_train, X_test, y_test = load_wisdm_2019(timesteps=100, step=40)

    assert X_train.shape == (89628, 100, 3)
    assert X_test.shape == (29876, 100, 3)

    assert y_train.shape == (89628, 1)
    assert y_test.shape == (29876, 1)


def test_load_plasticc():

    X_train, y_train, X_test, y_test = load_plasticc(timesteps=100, step=100)

    assert X_train.shape == (2991, 100, 6)
    assert X_test.shape == (998, 100, 6)

    assert y_train.shape == (2991, 1)
    assert y_test.shape == (998, 1)


def test_load_dataset():

    X_train, y_train, X_test, y_test, loss = load_dataset("plasticc")

    assert X_train.shape == (2991, 100, 6)
    assert X_test.shape == (998, 100, 6)

    assert y_train.shape == (2991, 3)
    assert y_test.shape == (998, 3)

    assert isinstance(WeightedLogLoss(), loss)
