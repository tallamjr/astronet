import json
import subprocess

import numpy as np
import pytest
import tensorflow as tf
from filelock import FileLock

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


@pytest.fixture(scope="session")
def get_fixt_UGRIZY_wZ(tmp_path_factory, worker_id, name="fixt_UGRIZY_wZ"):
    if not worker_id:
        # not executing in with multiple workers, just produce the data and let
        # pytest's fixture caching do its job
        return fixt_UGRIZY_wZ()

    # get the temp directory shared by all workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent

    fn = root_tmp_dir / "data.json"
    with FileLock(str(fn) + ".lock"):
        if fn.is_file():
            tdata = json.loads(fn.read_text())
            ldata = list(tdata)
            for item, index in enumerate(ldata):
                ldata[item] = tf.convert_to_tensor(index)
            data = tuple(ldata)
        else:
            tdata = fixt_UGRIZY_wZ()
            import pdb

            pdb.set_trace()
            ldata = list(tdata)
            for item, index in enumerate(ldata):
                ldata[item] = item.numpy()
            data = tuple(ldata)
            fn.write_text(json.dumps(data))
    return data


def fixt_UGRIZY_wZ():
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

    return test_ds, y_test_ds, test_input


@pytest.fixture(scope="session")
def get_fixt_UGRIZY_noZ(tmp_path_factory, worker_id, name="fixt_UGRIZY_noZ"):
    if not worker_id:
        # not executing in with multiple workers, just produce the data and let
        # pytest's fixture caching do its job
        return fixt_UGRIZY_noZ()

    # get the temp directory shared by all workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent

    fn = root_tmp_dir / "data.json"
    with FileLock(str(fn) + ".lock"):
        if fn.is_file():
            data = json.loads(fn.read_text())
        else:
            data = fixt_UGRIZY_noZ()
            fn.write_text(json.dumps(data))
    return data


def fixt_UGRIZY_noZ():
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


@pytest.fixture(scope="session")
def get_fixt_GR_noZ(tmp_path_factory, worker_id, name="fixt_GR_noZ"):
    if not worker_id:
        # not executing in with multiple workers, just produce the data and let
        # pytest's fixture caching do its job
        return fixt_GR_noZ()

    # get the temp directory shared by all workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent

    fn = root_tmp_dir / "data.json"
    with FileLock(str(fn) + ".lock"):
        if fn.is_file():
            data = json.loads(fn.read_text())
        else:
            data = fixt_GR_noZ()
            fn.write_text(json.dumps(data))
    return data


def fixt_GR_noZ():
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


# @pytest.fixture
# def fixt_UGRIZY_wZ_numpy(scope="session"):
#     """This fixture will only be available within the scope of TestPlots"""
#     X_test = np.load(
#         f"{asnwd}/data/plasticc/test_set/infer/X_test.npy",
#     )
#     y_test = np.load(
#         f"{asnwd}/data/plasticc/test_set/infer/y_test.npy",
#     )
#     Z_test = np.load(
#         f"{asnwd}/data/plasticc/test_set/infer/Z_test.npy",
#     )

#     inputs = [X_test, Z_test]

#     return X_test, y_test, Z_test, inputs
