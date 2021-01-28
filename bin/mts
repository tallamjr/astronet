#!/bin/bash
# Copyright 2020
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
#SBATCH --job-name=mts-arr              # Job name
#SBATCH --time=48:00:00                 # Time limit hrs:min:sec
#SBATCH --output=logs/%j.log            # Standard output and error log
#SBATCH --exclusive                     # Request exclusive access to a node
#SBATCH --cpus-per-task=24              # Number of CPUs per node
# Add a -e or --error with an error file name to separate output and error
# logs. Note this will ignore the --output option if used afterwards
## #SBATCH -e logs/%j.err
## #SBATCH -o logs/%j.out
#SBATCH --array=1-12
set -o pipefail -e
source $PWD/conf/astronet.conf
date
which python
# Test Imports
python -c "import astronet as asn; print(asn.__version__)"
python -c "import tensorflow as tf; print(tf.__version__)"

# Architecture
## Current options: {t2, snx}
ARCH=$1
echo "Using $ARCH architecture"

declare -a arr=(
                "ArabicDigits"
                "AUSLAN"
                "CharacterTrajectories"
                "CMUsubject16"
                "ECG"
                "JapaneseVowels"
                "KickvsPunch"
                "Libras"
                "NetFlow"
                "UWave"
                "Wafer"
                "WalkvsRun"
            )

echo "${arr[SLURM_ARRAY_TASK_ID-1]}"
dataset="${arr[SLURM_ARRAY_TASK_ID-1]}"
# Hyperparameter Optimisation
python $ASNWD/astronet/$ARCH/opt/hypertrain.py --dataset $dataset --epochs 50
# Train
# python $ASNWD/astronet/$ARCH/train.py --dataset $dataset --epochs 400
date
# Print the contents of this file to stdout from line 15 onwards
awk 'NR>15' $ASNWD/bin/mts
