import os
import pytest

from astronet.t2.utils import load_wisdm_2010, load_wisdm_2019


def test_load_wisdm_2010():

    X_train, y_train, X_val, y_val, X_test, y_test = load_wisdm_2010()

    assert X_train.shape == (21960, 200, 3)
    assert X_val.shape == (4114, 200, 3)
    assert X_test.shape == (1368, 200, 3)

    assert y_train.shape == (21960, 1)
    assert y_val.shape == (4114, 1)
    assert y_test.shape == (1368, 1)


@pytest.mark.skipif(os.getenv("CI") is not None, reason="Requires large 'phone.df' file")
def test_load_wisdm_2019():

    X_train, y_train, X_val, y_val, X_test, y_test = load_wisdm_2019()

    assert X_train.shape == (95603, 100, 3)
    assert X_val.shape == (17924, 100, 3)
    assert X_test.shape == (5973, 100, 3)

    assert y_train.shape == (95603, 1)
    assert y_val.shape == (17924, 1)
    assert y_test.shape == (5973, 1)


def test_plasticc():
    pass
