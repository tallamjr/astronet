# `astronet`

                                                      .  '  *   .  . '
                                                          .  * * -+-
                                                      .    * .    '  *
                                                          * .  ' .  .
                                                       *   *  .   .
                                                         '   *
                  _____                         _____
    ______ _________  /___________________________  /_
    _  __ `/_  ___/  __/_  ___/  __ \_  __ \  _ \  __/
    / /_/ /_(__  )/ /_ _  /   / /_/ /  / / /  __/ /_
    \__,_/ /____/ \__/ /_/    \____//_/ /_/\___/\__/


![Test Suite with Code Coverage](https://github.com/tallamjr/astronet/workflows/Test%20Suite%20with%20Code%20Coverage/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/tallamjr/astronet/branch/master/graph/badge.svg?token=X2RP4DC3K1)](https://codecov.io/gh/tallamjr/astronet)

`astronet` is a package to classify Astrophysical transients using Deep Learning methods

## Experimental Roadmap

- ~~Time Transformer [`t2`] (WISDM: Human-Activity-Recognition)~~
- ~~`transnova` (SPCC)~~
- ~~Astrophysical Transient Transformer [`att`] (PLAsTiCC)~~

**Update 20201211**

- ~~Time-Series Transformer [`t2`] (WISDM: Human-Activity-Recognition) & PLAsTiCC and other MVTS~~
    ~~--> May change overall name to `sncoder` for `Supernova-Encoder`. Will see.~~
- ~~Inception-Time for Supernova [`convSNE`] --> All of above including MVTS~~

**Update 20201220**

- Timeseries Transformer [`t2`]
    Multi-headed attention encoder inspired by Transformers but with convolutions embedding instead
- snXception [`snX`]
    Adaptation of Xception networks with 1D Depthwise-Separable convolutions in place of the 2D
    version
- Supernova-Similarity-Search with Siamese Networks [`s3`]
    Utilisation of either `t2` or `snX` networks depending on performance, to achieve one-shot
    classification, by comparing similarity of hamming distance or Locality Sensitive Hashing (LSH)
    to compare similarity.

## System Design

## Running on `hypatia`

Within `bin` are the relevant slurm scripts for running jobs on the cluster.
Below is an example of the `t2` script to run the T2 analysis (file snapshot as
of 20201211)

```bash
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
#SBATCH --job-name=plasticc             # Job name
#SBATCH --time=48:00:00                 # Time limit hrs:min:sec
#SBATCH --output=logs/%j.log            # Standard output and error log
#SBATCH --exclusive                     # Request exclusive access to a node
# Add a -e or --error with an error file name to separate output and error
# logs. Note this will ignore the --output option if used afterwards
## #SBATCH -e logs/%j.err
## #SBATCH -o logs/%j.out
set -o pipefail -e

source $PWD/conf/astronet.conf
date
which python
# Test Imports
python -c "import astronet as asn; print(asn.__version__)"
python -c "import tensorflow as tf; print(tf.__version__)"
# Hyperparameter Optimisation
python $ASNWD/astronet/t2/opt/hypertrain.py --dataset "plasticc" --epochs 40 --batch-size 256
# Train
python $ASNWD/astronet/t2/train.py --dataset "plasticc" --epochs 200 --batch-size 256
date
# Print the contents of this file to stdout from line 15 onwards
awk 'NR>15' $ASNWD/bin/t2
```

To run this, ensure that the anaconda environment is set (a simple test for this
is to run `which python`), then if everything is in place, one can run:
```
$ sbatch bin/t2
```
This will send a job to the scheduler. The logs for the specified job would be
in the `astronet/logs` directory within `<JOB-ID>.log` file

**NOTE** The `$ASNWD` is set via the `conf/astronet.conf` file. This is where
the home directory of the `astronet` repository should be set.

### Papers of Interest

- [WISDM Smartphone and Smartwatch Activity and Biometrics Dataset AKA WISDM-2019]("./resources/papers/WISDM-dataset-description.pdf)
- [Deep Learning for Sensor-based Human Activity Recognition: Overview, Challenges and Opportunities]("./resources/papers/2001.07416.pdf")
- [Smartphone Location Recognition: A Deep Learning-Based Approach]("./resources/papers/sensors-20-00214-v2.pdf")
- [Deep Learning Models for Real-time Human Activity Recognition with Smartphones]("./resources/papers/Wan2020_Article_DeepLearningModelsForReal-time.pdf)
- [A Lightweight Deep Learning Model for Human Activity Recognition on Edge Devices Recognition on Edge Devices]("./resources/papers/1-s2.0-S1877050920307559-main.pdf)

- [Human Activity Recognition from Wearable Sensor Data Using Self-Attention](https://arxiv.org/pdf/2003.09018.pdf)
- [On Attention Models for Human Activity Recognition](https://arxiv.org/pdf/1805.07648.pdf)

### MTS Benchmark Results
|                       |        t2 |      snX | MLP        | FCN        | ResNet     | Encoder    | MCNN      | t-LeNet    | MCDCNN     | Time-CNN   | TWIESN     |
|:----------------------|----------:|---------:|:-----------|:-----------|:-----------|:-----------|:----------|:-----------|:-----------|:-----------|:-----------|
| ArabicDigits          |  94.0909  | 0.001    | 96.9(0.2)  | 99.4(0.1)  | 99.6(0.1)  | 98.1(0.1)  | 10.0(0.0) | 10.0(0.0)  | 95.9(0.2)  | 95.8(0.3)  | 85.3(1.4)  |
| AUSLAN                |   1.05263 | 8.33684  | 93.3(0.5)  | 97.5(0.4)  | 97.4(0.3)  | 93.8(0.5)  | 1.1(0.0)  | 1.1(0.0)   | 85.4(2.7)  | 72.6(3.5)  | 72.4(1.6)  |
| CharacterTrajectories |   4.76935 | 9.64425  | 96.9(0.2)  | 99.0(0.1)  | 99.0(0.2)  | 97.1(0.2)  | 5.4(0.8)  | 6.7(0.0)   | 93.8(1.7)  | 96.0(0.8)  | 92.0(1.3)  |
| CMUsubject16          | 100       | 7.24138  | 60.0(16.9) | 100.0(0.0) | 99.7(1.1)  | 98.3(2.4)  | 53.1(4.4) | 51.0(5.3)  | 51.4(5.0)  | 97.6(1.7)  | 89.3(6.8)  |
| ECG                   |  78       | 3.3      | 74.8(16.2) | 87.2(1.2)  | 86.7(1.3)  | 87.2(0.8)  | 67.0(0.0) | 67.0(0.0)  | 50.0(17.9) | 84.1(1.7)  | 73.7(2.3)  |
| JapaneseVowels        |  95.4054  | 2.10811  | 97.6(0.2)  | 99.3(0.2)  | 99.2(0.3)  | 97.6(0.6)  | 9.2(2.5)  | 23.8(0.0)  | 94.4(1.4)  | 95.6(1.0)  | 96.5(0.7)  |
| KickvsPunch           |  50       | 4        | 61.0(12.9) | 54.0(13.5) | 51.0(8.8)  | 61.0(9.9)  | 54.0(9.7) | 50.0(10.5) | 56.0(8.4)  | 62.0(6.3)  | 67.0(14.2) |
| Libras                |   6.66667 | 0.666667 | 78.0(1.0)  | 96.4(0.7)  | 95.4(1.1)  | 78.3(0.9)  | 6.7(0.0)  | 6.7(0.0)   | 65.1(3.9)  | 63.7(3.3)  | 79.4(1.3)  |
| NetFlow               |  77.9026  | 7.79026  | 55.0(26.1) | 89.1(0.4)  | 62.7(23.4) | 77.7(0.5)  | 77.9(0.0) | 72.3(17.6) | 63.0(18.2) | 89.0(0.9)  | 94.5(0.4)  |
| UWave                 |  84.5255  | 5.76204  | 90.1(0.3)  | 93.4(0.3)  | 92.6(0.4)  | 90.8(0.4)  | 12.5(0.0) | 12.5(0.0)  | 84.5(1.6)  | 85.9(0.7)  | 75.4(6.3)  |
| Wafer                 |  89.3973  | 8.93973  | 89.4(0.0)  | 98.2(0.5)  | 98.9(0.4)  | 98.6(0.2)  | 89.4(0.0) | 89.4(0.0)  | 65.8(38.1) | 94.8(2.1)  | 94.9(0.6)  |
| WalkvsRun             | 100       | 2.5      | 70.0(15.8) | 100.0(0.0) | 100.0(0.0) | 100.0(0.0) | 75.0(0.0) | 60.0(24.2) | 45.0(25.8) | 100.0(0.0) | 94.4(9.1)  |
