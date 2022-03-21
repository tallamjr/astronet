import subprocess

import numpy as np
import pytest

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd

ISA = subprocess.run(
    "uname -m",
    check=True,
    capture_output=True,
    shell=True,
    text=True,
).stdout.strip()

SKIP_IF_M1 = pytest.mark.skipif(ISA == "arm64", reason="Error on arm-m1")


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

    inputs = [X_test, Z_test]

    return X_test, y_test, Z_test, inputs
