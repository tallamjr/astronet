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
import shutil
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from astronet.utils import astronet_logger, save_avocado_training_set

try:
    log = astronet_logger(__file__)
    log.info("=" * shutil.get_terminal_size((80, 20))[0])
    log.info(f"File Path: {Path(__file__).absolute()}")
    log.info(f"Parent of Directory Path: {Path().absolute().parent}")
except:
    print("Seems you are running from a notebook...")
    __file__ = f"{Path().resolve().parent}/sbin/plasticc-test-set.py"

np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.2f}".format})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class ProcessTrainingSet(object):
    # TODO: Update docstrings
    def __init__(self, filename):
        self.filename = filename

    def __call__(self):

        save_avocado_training_set(batch_filename=self.filename, redshift=True)
        print(f"{self.filename} COMPLETED")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process Avocado Augmented Training Set"
    )

    parser.add_argument(
        "-f",
        "--file",
        default="avo_aug_1_chunk_0",
        help="Choose which dataset to process",
    )

    try:
        args = parser.parse_args()
        argsdict = vars(args)
    except KeyError:
        parser.print_help()
        sys.exit(0)

    filename = args.file

    process_training_set = ProcessTrainingSet(filename=filename)
    process_training_set()
