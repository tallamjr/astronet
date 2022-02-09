## Papers of Interest

- [WISDM Smartphone and Smartwatch Activity and Biometrics Dataset AKA WISDM-2019]("./resources/papers/WISDM-dataset-description.pdf)
- [Deep Learning for Sensor-based Human Activity Recognition: Overview, Challenges and Opportunities]("./resources/papers/2001.07416.pdf")
- [Smartphone Location Recognition: A Deep Learning-Based Approach]("./resources/papers/sensors-20-00214-v2.pdf")
- [Deep Learning Models for Real-time Human Activity Recognition with Smartphones]("./resources/papers/Wan2020_Article_DeepLearningModelsForReal-time.pdf)
- [A Lightweight Deep Learning Model for Human Activity Recognition on Edge Devices Recognition on Edge Devices]("./resources/papers/1-s2.0-S1877050920307559-main.pdf)

- [Human Activity Recognition from Wearable Sensor Data Using Self-Attention](https://arxiv.org/pdf/2003.09018.pdf)
- [On Attention Models for Human Activity Recognition](https://arxiv.org/pdf/1805.07648.pdf)


## Experimental Roadmap

- ~~Time Transformer [`t2`] (WISDM: Human-Activity-Recognition)~~
- ~~`transnova` (SPCC)~~
- ~~Astrophysical Transient Transformer [`att`] (PLAsTiCC)~~

**Update 20201211**

- ~~Time-Series Transformer [`t2`] (WISDM: Human-Activity-Recognition) & PLAsTiCC and other MVTS~~
    ~~--> May change overall name to `sncoder` for `Supernova-Encoder`. Will see.~~
- ~~Inception-Time for Supernova [`convSNE`] --> All of above including MVTS~~

**Update 20201220**

- ~~Timeseries Transformer [`t2`]~~
    ~~Multi-headed attention encoder inspired by Transformers but with convolutions embedding instead~~
- ~~snXception [`snX`]~~
    ~~Adaptation of Xception networks with 1D Depthwise-Separable convolutions in place of the 2D~~
    ~~version~~
- ~~Supernova-Similarity-Search with Siamese Networks [`s3`]~~
    ~~Utilisation of either `t2` or `snX` networks depending on performance, to achieve one-shot~~
    ~~classification, by comparing similarity of hamming distance or Locality Sensitive Hashing (LSH)~~
    ~~to compare similarity.~~

**Update 20211209**

- Timeseries Transformer [`t2`]
    Multi-headed attention encoder inspired by Transformers but with convolutions embedding instead
- Astronomical Transients + Xception [`atx`]
    Adaptation of Xception networks with 1D Depthwise-Separable convolutions in place of the 2D version
    - https://github.com/tensorflow/tensorflow/issues/36935
- Deployment into FINK
    Apply `t2` to real ZTF alert data.
- [TIME PERMITTING] Extend `t2` to have probabilistic layers for distributions over weights instead
    of point estimates.

## System Design

* [Scheduling Jobs in a Super Computer Cluster Using Github Actions](https://thepaulo.medium.com/scheduling-jobs-in-a-super-computer-cluster-using-github-actions-2ee83405cb69)

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

