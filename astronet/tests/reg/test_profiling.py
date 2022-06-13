import os
import subprocess

import pytest

from astronet.utils import astronet_logger

log = astronet_logger(__file__)


@pytest.mark.skipif(
    os.getenv("CI") is not None, reason="Produces report to be checked locally"
)
def test_kernprof():
    prof = "load_run_lpa.py"
    _ = subprocess.run(
        f"kernprof -lv {prof} 2>&1 | tee {prof}.stdout.txt",
        check=True,
        capture_output=True,
        shell=True,
        executable="/opt/homebrew/bin/bash",
        text=True,
    ).stdout.strip()

    log.info(f"KERNPROF DONE: Saved to {prof}.lnprof")

    out = subprocess.run(
        f"python -m line_profiler {prof}.lprof &>> {prof}.stdout.txt",
        check=True,
        capture_output=True,
        shell=True,
        executable="/opt/homebrew/bin/bash",
        text=True,
    ).stdout.strip()

    log.info(out)
    assert True
