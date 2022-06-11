#!/opt/homebrew/bin/bash
# Copyright 2022
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
set -o pipefail -e
SECONDS=0 # https://stackoverflow.com/a/8903280/4521950
python -c "import astronet as asn; print(asn.__version__)"

# Tensorflow
export TF_CPP_MIN_LOG_LEVEL=2

architecture=$1
dataset="plasticc"

if [ $architecture == "atx" ]; then
    model="9887359-1641295475-0.1.dev943+gc9bafac.d20220104"    # atx
else
    model="1619624444-0.1.dev765+g7c90cbb.d20210428"            # t2
    # model="9901958-1652622376-0.5.1.dev11+g733e01d"    # tinho
fi

python $ASNWD/astronet/t2/compress.py \
    --architecture $architecture \
    --dataset $dataset \
    --model $model \
    --redshift "" # With or without redshift. Remove argument if without

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
