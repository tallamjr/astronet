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

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd


def chunker(seq, size):
    """https://stackoverflow.com/a/434328/4521950"""
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def read_large_file(filename):
    df = pd.read_csv(filename)
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process PLAsTiCC Test Set in Chunks")

    parser.add_argument(
        "-n",
        "--num_objects",
        default=10000,
        type=int,
        help="Choose which dataset to process",
    )

    parser.add_argument(
        "-f",
        "--file",
        default=f"{asnwd}/data/plasticc/test_set/plasticc_test_lightcurves_01.csv",
        help="Choose which dataset to process",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default=f"{asnwd}/data/plasticc/test_set/",
        help="Choose which dataset to process",
    )

    try:
        args = parser.parse_args()
        argsdict = vars(args)
        print(argsdict)
    except KeyError:
        parser.print_help()
        sys.exit(0)

    filename = args.file
    num_objects_per_chunk = args.num_objects
    output_dir = args.output_dir

    df = read_large_file(filename)

    object_list = list(np.unique(df["object_id"]))

    for idx, objects in enumerate(chunker(object_list, num_objects_per_chunk)):
        dfs = df[df["object_id"].isin(objects)]
        print(idx, dfs.shape)
        dfs.to_csv(f"{output_dir}{Path(filename).stem}_chunk_{idx}.csv")
