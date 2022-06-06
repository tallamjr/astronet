import subprocess

import numpy as np
import pytest
import tensorflow as tf

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.constants import LOCAL_DEBUG

ISA = subprocess.run(
    "uname -m",
    check=True,
    capture_output=True,
    shell=True,
    text=True,
).stdout.strip()

SKIP_IF_M1 = pytest.mark.skipif(ISA == "arm64", reason="Error on arm-m1")

BATCH_SIZE = 64


@pytest.fixture
def fixt_numpy(scope="session"):
    """This fixture will only be available within the scope of TestPlots"""
    X_test = np.load(
        f"{asnwd}/data/plasticc/test_set/infer/X_test.npy",
    )
    y_test = np.load(
        f"{asnwd}/data/plasticc/test_set/infer/y_test.npy",
    )
    Z_test = np.load(
        f"{asnwd}/data/plasticc/test_set/infer/Z_test.npy",
    )

    inputs = [X_test, Z_test]

    return X_test, y_test, Z_test, inputs


@pytest.fixture
def fixt(scope="session"):
    """This fixture will only be available within the scope of TestPlots"""
    X_test = np.load(
        f"{asnwd}/data/plasticc/test_set/infer/X_test.npy",
    )
    y_test = np.load(
        f"{asnwd}/data/plasticc/test_set/infer/y_test.npy",
    )
    Z_test = np.load(
        f"{asnwd}/data/plasticc/test_set/infer/Z_test.npy",
    )

    test_input = [X_test, Z_test]

    test_ds = (
        tf.data.Dataset.from_tensor_slices(
            ({"input_1": test_input[0], "input_2": test_input[1]}, y_test)
        )
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    y_test_ds = (
        tf.data.Dataset.from_tensor_slices(y_test)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    if LOCAL_DEBUG is not None:
        print("LOCAL_DEBUG set, reducing dataset size...")
        test_ds = test_ds.take(300)
        y_test_ds = y_test_ds.take(300)

    return test_ds, y_test_ds


@pytest.fixture
def fixtnoz(scope="session"):
    """This fixture will only be available within the scope of TestPlots"""
    X_test = np.load(
        f"{asnwd}/data/plasticc/test_set/infer/X_test.npy",
    )
    y_test = np.load(
        f"{asnwd}/data/plasticc/test_set/infer/y_test.npy",
    )

    test_input = X_test

    test_ds = (
        tf.data.Dataset.from_tensor_slices((test_input, y_test))
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    y_test_ds = (
        tf.data.Dataset.from_tensor_slices(y_test)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    if LOCAL_DEBUG is not None:
        print("LOCAL_DEBUG set, reducing dataset size...")
        test_ds = test_ds.take(300)
        y_test_ds = y_test_ds.take(300)

    return test_ds, y_test_ds


@pytest.fixture
def fixtztf(scope="session"):
    """This fixture will only be available within the scope of TestPlots"""
    X_test = np.load(
        f"{asnwd}/data/plasticc/test_set/infer/X_test.npy",
    )
    y_test = np.load(
        f"{asnwd}/data/plasticc/test_set/infer/y_test.npy",
    )

    X_test = X_test[:, :, 0:3:2]
    test_input = X_test

    test_ds = (
        tf.data.Dataset.from_tensor_slices((test_input, y_test))
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    y_test_ds = (
        tf.data.Dataset.from_tensor_slices(y_test)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    if LOCAL_DEBUG is not None:
        print("LOCAL_DEBUG set, reducing dataset size...")
        test_ds = test_ds.take(300)
        y_test_ds = y_test_ds.take(300)

    return test_ds, y_test_ds
