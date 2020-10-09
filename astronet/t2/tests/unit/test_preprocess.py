from astronet.t2.preprocess import one_hot_encode
from astronet.t2.utils import load_wisdm_2010, load_wisdm_2019


def test_one_hot_encode():

    X_train, y_train, X_val, y_val, X_test, y_test = load_wisdm_2010()

    assert y_train.shape == (21960, 1)
    assert y_val.shape == (4114, 1)
    assert y_test.shape == (1368, 1)

    # One hot encode y
    enc, y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test)

    assert y_train.shape == (21960, 6)
    assert y_val.shape == (4114, 6)
    assert y_test.shape == (1368, 6)

    del enc, X_train, y_train, X_val, y_val, X_test, y_test

    X_train, y_train, X_val, y_val, X_test, y_test = load_wisdm_2019()

    assert y_train.shape == (95603, 1)
    assert y_val.shape == (17924, 1)
    assert y_test.shape == (5973, 1)

    # One hot encode y
    enc, y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test)

    assert y_train.shape == (95603, 18)
    assert y_val.shape == (17924, 18)
    assert y_test.shape == (5973, 18)
