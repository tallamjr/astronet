# Copyright 2020 - 2022
# Author: Tarek Allam Jr.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess

import pytest

from astronet.utils import astronet_logger

log = astronet_logger(__file__)


@pytest.mark.skipif(
    os.getenv("CI") is not None, reason="Produces report to be checked locally"
)
@pytest.mark.parametrize(
    ("dataset"),
    (
        ("plasticc",),
        ("ztf",),
    ),
)
def test_kernprof(dataset):
    prof = f"{dataset}-load-run-lpa.py"
    _ = subprocess.run(
        f"kernprof -lv {prof} 2>&1 | tee {prof}.stdout.txt",
        check=True,
        capture_output=True,
        shell=True,
        executable="/opt/homebrew/bin/bash",
        text=True,
    ).stdout.strip()

    log.info(f"KERNPROF DONE: Saved to {prof}.lnprof")

    proc = subprocess.run(
        f"python -m line_profiler {prof}.lprof &>> {prof}.stdout.txt",
        check=True,
        capture_output=True,
        shell=True,
        executable="/opt/homebrew/bin/bash",
        text=True,
    ).stdout.strip()

    out = proc.stdout.strip()
    log.info(out)

    result = proc.returncode

    assert result == 0  # Successful exit code
