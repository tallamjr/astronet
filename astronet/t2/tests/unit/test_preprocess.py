import os
import pytest

from astronet.t2.preprocess import *
from astronet.t2.utils import load_wisdm_2010, load_wisdm_2019


def test_one_hot_encode():

    X_train, y_train, X_val, y_val, X_test, y_test = load_wisdm_2010()

    assert y_train.shape == (21960, 1)
    assert y_val.shape == (4114, 1)
    assert y_test.shape == (1368, 1)

    # One hot encode y
    enc, y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test)

    assert y_train.shape == (21960, 6)
    assert y_train.shape[1] == 6
    assert y_val.shape == (4114, 6)
    assert y_test.shape == (1368, 6)

    del enc, X_train, y_train, X_val, y_val, X_test, y_test


@pytest.mark.skipif(os.getenv("CI") is not None, reason="Requires large 'phone.df' file")
def test_one_hot_encode_local():

    X_train, y_train, X_val, y_val, X_test, y_test = load_wisdm_2019()

    assert y_train.shape == (95603, 1)
    assert y_val.shape == (17924, 1)
    assert y_test.shape == (5973, 1)

    # One hot encode y
    enc, y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test)

    assert y_train.shape == (95603, 18)
    assert y_val.shape == (17924, 18)
    assert y_test.shape == (5973, 18)


def test_plasticc_fit_2d_gp():
    pass


def test_plasticc_predict_2d_gp():
    pass
