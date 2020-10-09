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


`astronet` is a package to classify Astrophysical transients using Deep Learning methods

## Experimental Roadmap

- Time Transformer [`t2`] (WISDM: Human-Activity-Recognition)
- `transnova` (SPCC)
- Astrophysical Transient Transformer [`att`] (PLAsTiCC)

## System Design

## Running on `hypatia`

Within `bin` are the relevant slurm scripts for running jobs on the cluster.
Below is an example of the `t2` script to run the T2 analysis (file snapshot as
of 20201009)

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
#SBATCH --job-name=t2-wisdm             # Job name
#SBATCH --time=168:00:00                # Time limit hrs:min:sec
#SBATCH --output=logs/%j.log            # Standard output and error log
#SBATCH --exclusive                     # Request exclusive access to a node
# Add a -e or --error with an error file name to separate output and error
# logs. Note this will ignore the --output option if used afterwards
## #SBATCH -e logs/%j.err
## #SBATCH -o logs/%j.out
set -o pipefail -e
hmsg="Help message..."
# Show help if no arguments is given
# if [[ $1 == "" ]]; then
#   echo -e $hmsg
#   exit 1
# fi
# cd $(cd "`dirname "$0"`"/..; pwd)
source $PWD/conf/astronet.conf
date
which python
# Test imports
python -c "import astronet as asn; print(asn.__version__)"
python -c "import tensorflow as tf; print(tf.__version__)"
# Hyperparameter Optimisation
python $ASNWD/astronet/t2/opt/hypertrain.py
# Train
python $ASNWD/astronet/t2/train.py
date
```

To run this, ensure that the anaconda environment is set (a simple test for this
is to run `which python`), then if everything is in place, one can run:
```
$ sbatch bin/t2
```
This will send a job to the scheduler. The logs for the specified job would be
in the `astronet/logs` directory within `<JOB-ID>.log` file

### Papers of Interest

- [WISDM Smartphone and Smartwatch Activity and Biometrics Dataset AKA WISDM-2019]("./resources/papers/WISDM-dataset-description.pdf)
- [Deep Learning for Sensor-based Human Activity Recognition: Overview, Challenges and Opportunities]("./resources/papers/2001.07416.pdf")
- [Smartphone Location Recognition: A Deep Learning-Based Approach]("./resources/papers/sensors-20-00214-v2.pdf")
- [Deep Learning Models for Real-time Human Activity Recognition with Smartphones]("./resources/papers/Wan2020_Article_DeepLearningModelsForReal-time.pdf)
- [A Lightweight Deep Learning Model for Human Activity Recognition on Edge Devices Recognition on Edge Devices]("./resources/papers/1-s2.0-S1877050920307559-main.pdf)

- [Human Activity Recognition from Wearable Sensor Data Using Self-Attention](https://arxiv.org/pdf/2003.09018.pdf)
- [On Attention Models for Human Activity Recognition](https://arxiv.org/pdf/1805.07648.pdf)
