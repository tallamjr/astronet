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

    X_train, y_train, X_val, y_val, X_test, y_test = load_wisdm_2019()

    ((19120, 200, 3), (19120, 1), (3585, 200, 3), (3585, 1), (1195, 200, 3), (1195, 1))

    assert X_train.shape == (19120, 200, 3)
    assert X_val.shape == (17924, 200, 3)
    assert X_test.shape == (5973, 200, 3)

    assert y_train.shape == (19120, 1)
    assert y_val.shape == (3585, 1)
    assert y_test.shape == (1195, 1)


def test_load_plasticc():

    X_train, y_train, X_val, y_val, X_test, y_test = load_plasticc(timesteps=20, step=20)

    assert X_train.shape == (15959, 20, 6)
    assert X_val.shape == (2992, 20, 6)
    assert X_test.shape == (997, 20, 6)

    assert y_train.shape == (15959, 1)
    assert y_val.shape == (2992, 1)
    assert y_test.shape == (997, 1)
