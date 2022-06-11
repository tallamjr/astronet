import os
import subprocess

import numpy as np
import pytest
import tensorflow as tf

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.constants import LOCAL_DEBUG
from astronet.utils import astronet_logger

log = astronet_logger(__file__)


@pytest.mark.skipif(
    os.getenv("CI") is not None, reason="Produces report to be checked locally"
)
def test_kernprof():
    prof = "lnprofile.py"
    _ = subprocess.run(
        f"kernprof -lv {prof}",
        check=True,
        capture_output=True,
        shell=True,
        text=True,
    ).stdout.strip()

    log.info(f"KERNPROF DONE: Saved to {prof}.lnprof")

    out = subprocess.run(
        f"python -m line_profiler {prof}.lprof",
        check=True,
        capture_output=True,
        shell=True,
        text=True,
    ).stdout.strip()

    log.info(out)
    assert True
