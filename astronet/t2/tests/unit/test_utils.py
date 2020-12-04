import os
import pytest

from astronet.t2.utils import load_wisdm_2010, load_wisdm_2019, load_plasticc


def test_load_wisdm_2010():

    X_train, y_train, X_val, y_val, X_test, y_test = load_wisdm_2010(timesteps=200, step=200)

    assert X_train.shape == (4392, 200, 3)
    assert X_val.shape == (823, 200, 3)
    assert X_test.shape == (274, 200, 3)

    assert y_train.shape == (4392, 1)
    assert y_val.shape == (823, 1)
    assert y_test.shape == (274, 1)


@pytest.mark.skipif(os.getenv("CI") is not None, reason="Requires large 'phone.df' file")
def test_load_wisdm_2019():

    X_train, y_train, X_val, y_val, X_test, y_test = load_wisdm_2019(timesteps=100, step=40)

    assert X_train.shape == (95603, 100, 3)
    assert X_val.shape == (17924, 100, 3)
    assert X_test.shape == (5973, 100, 3)

    assert y_train.shape == (95603, 1)
    assert y_val.shape == (17924, 1)
    assert y_test.shape == (5973, 1)


def test_load_plasticc():

    X_train, y_train, X_val, y_val, X_test, y_test = load_plasticc(timesteps=100, step=100)

    assert X_train.shape == (3191, 100, 6)
    assert X_val.shape == (598, 100, 6)
    assert X_test.shape == (199, 100, 6)

    assert y_train.shape == (3191, 1)
    assert y_val.shape == (598, 1)
    assert y_test.shape == (199, 1)
