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
import platform
from pathlib import Path

try:
    ASTRONET_WORKING_DIRECTORY = f"{os.environ['ASNWD']}"
except Exception as e:
    print(f"Environment variable not set:{e}.\nDefining relative to constants.py file")
    ASTRONET_WORKING_DIRECTORY = Path(__file__).absolute().parent.parent

SYSTEM = platform.system()
# Linux: Linux
# Mac: Darwin
# Windows: Windows
LOCAL_DEBUG = os.environ.get("LOCAL_DEBUG")
# ================================================== PLASTICC ==================================== #
PLASTICC_CLASS_MAPPING = {
    90: "SNIa",
    67: "SNIa-91bg",
    52: "SNIax",
    42: "SNII",
    62: "SNIbc",
    95: "SLSN-I",
    15: "TDE",
    64: "KN",
    88: "AGN",
    92: "RRL",
    65: "M-dwarf",
    16: "EB",
    53: "Mira",
    6: "$\mu$-Lens-Single",
}

PLASTICC_WEIGHTS_DICT = {
    6: 1 / 18,
    15: 1 / 9,
    16: 1 / 18,
    42: 1 / 18,
    52: 1 / 18,
    53: 1 / 18,
    62: 1 / 18,
    64: 1 / 9,
    65: 1 / 18,
    67: 1 / 18,
    88: 1 / 18,
    90: 1 / 18,
    92: 1 / 18,
    95: 1 / 18,
    99: 1 / 19,
    1: 1 / 18,
    2: 1 / 18,
    3: 1 / 18,
}

# ===================================================== LSST ==================================== #
LSST_FILTER_MAP = {
    0: "lsstu",
    1: "lsstg",
    2: "lsstr",
    3: "lssti",
    4: "lsstz",
    5: "lssty",
}

# Central passbands wavelengths
LSST_PB_WAVELENGTHS = {
    "lsstu": 3685.0,
    "lsstg": 4802.0,
    "lsstr": 6231.0,
    "lssti": 7542.0,
    "lsstz": 8690.0,
    "lssty": 9736.0,
}

LSST_PB_COLORS = {
    "lsstu": "#984ea3",  # Purple: https://www.color-hex.com/color/984ea3
    "lsstg": "#4daf4a",  # Green: https://www.color-hex.com/color/4daf4a
    "lsstr": "#e41a1c",  # Red: https://www.color-hex.com/color/e41a1c
    "lssti": "#377eb8",  # Blue: https://www.color-hex.com/color/377eb8
    "lsstz": "#ff7f00",  # Orange: https://www.color-hex.com/color/ff7f00
    "lssty": "#e3c530",  # Yellow: https://www.color-hex.com/color/e3c530
}


# ===================================================== ZTF ==================================== #
ZTF_FILTER_MAP = {1: "ztfg", 2: "ztfr", 3: "ztfi"}

ZTF_FILTER_MAP_COLORS = {
    1: "#4daf4a",  # Green: https://www.color-hex.com/color/4daf4a
    2: "#e41a1c",  # Red: https://www.color-hex.com/color/e41a1c
    3: "#377eb8",  # Blue: https://www.color-hex.com/color/377eb8
}

# Central passbands wavelengths --> λmean
# http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=Palomar&gname2=ZTF&asttype=
ZTF_PB_WAVELENGTHS = {
    "ztfg": 4804.79,
    "ztfr": 6436.92,
    "ztfi": 7968.22,
}

ZTF_PB_COLORS = {
    "ztfg": "#4daf4a",  # Green: https://www.color-hex.com/color/4daf4a
    "ztfr": "#e41a1c",  # Red: https://www.color-hex.com/color/e41a1c
    "ztfi": "#377eb8",  # Blue: https://www.color-hex.com/color/377eb8
}
