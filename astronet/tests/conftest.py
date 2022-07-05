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

import inspect
import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from filelock import FileLock

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.utils import astronet_logger

log = astronet_logger(__file__)

ISA = subprocess.run(
    "uname -m",
    check=True,
    capture_output=True,
    shell=True,
    text=True,
).stdout.strip()

SKIP_IF_M1 = pytest.mark.skipif(ISA == "arm64", reason="Error on arm-m1")

BATCH_SIZE = 2048


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def pandas_encoder(obj):
    # TODO: Reshape required to fix ValueError: Must pass 2-d input. shape=(869864, 100, 6)
    # Refs:
    # - https://stackoverflow.com/a/32034565/4521950
    # - https://stackoverflow.com/a/32838859/4521950
    # - https://stackoverflow.com/a/44752209/4521950
    log.critical(f"{inspect.stack()[0].function} -- Not Fully Implemented Yet")
    return pd.DataFrame(obj).to_json(orient="values")


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
            data = json.loads(fn.read_text())
            X_test = np.asarray(data["X_test"])
            y_test = np.asarray(data["y_test"])
            Z_test = np.asarray(data["Z_test"])
        else:
            X_test, y_test, Z_test = fixt_UGRIZY_wZ()
            fn.write_text(
                json.dumps(
                    {"X_test": X_test, "y_test": y_test, "Z_test": Z_test},
                    cls=NumpyEncoder,
                    # default=pandas_encoder,
                )
            )
    return X_test, y_test, Z_test


def fixt_UGRIZY_wZ():
    """This fixture will only be available within the scope of TestPlots"""
    X_test = np.load(f"{asnwd}/data/plasticc/processed/X_test.npy")
    Z_test = np.load(f"{asnwd}/data/plasticc/processed/Z_test.npy")
    y_test = np.load(f"{asnwd}/data/plasticc/processed/y_test.npy")

    return X_test, y_test, Z_test
