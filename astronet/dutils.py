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

import subprocess
import warnings
import zipfile
from typing import Dict

import pandas as pd
from fink_client.visualisation import extract_field
from tensorflow.python.ops.numpy_ops import np_config

# 'SettingWithCopyWarning' in Pandas: https://bit.ly/3mv3fhw
pd.options.mode.chained_assignment = None  # default='warn'

warnings.filterwarnings("ignore")

np_config.enable_numpy_behavior()


def extract_fields(alert: Dict):
    # Get flux and error
    magpsf = extract_field(alert, "magpsf")
    sigmapsf = extract_field(alert, "sigmapsf")

    jd = extract_field(alert, "jd")

    # For rescaling dates to start at 0 --> 30
    # dates = np.array([jd[0] - i for i in jd])

    # FINK candidate ID (int64)
    candid = alert["candid"]

    # filter bands
    fid = extract_field(alert, "fid")

    return (magpsf, sigmapsf, jd, candid, fid)


def check_size(filepath):
    du = subprocess.run(
        f"du -sh {filepath} | awk '{{print $1}}'",
        check=True,
        capture_output=True,
        shell=True,
        text=True,
    ).stdout
    return du


def zippify(file_path: str, zipped_name: str):

    if zipped_name is None:
        zipped_name = file_path + ".zip"

    with zipfile.ZipFile(
        zipped_name,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.write(file_path)

    return zipped_name
