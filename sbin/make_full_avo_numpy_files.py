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

import numpy as np

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd

batch_filename = [
    "avo_aug_1_chunk_0",
    "avo_aug_1_chunk_1",
    "avo_aug_1_chunk_2",
    "avo_aug_1_chunk_3",
    "avo_aug_1_chunk_4",
    "avo_aug_1_chunk_5",
    "avo_aug_1_chunk_6",
    "avo_aug_1_chunk_7",
    "avo_aug_1_chunk_8",
    "avo_aug_1_chunk_9",
    "avo_aug_1_chunk_10",
    "avo_aug_1_chunk_11",
    "avo_aug_1_chunk_12",
    "avo_aug_1_chunk_13",
    "avo_aug_1_chunk_14",
    "avo_aug_1_chunk_15",
    "avo_aug_1_chunk_16",
    "avo_aug_1_chunk_17",
    "avo_aug_1_chunk_18",
    "avo_aug_1_chunk_19",
    "avo_aug_1_chunk_20",
    "avo_aug_1_chunk_21",
    "avo_aug_1_chunk_22",
    "avo_aug_1_chunk_23",
    "avo_aug_1_chunk_24",
    "avo_aug_1_chunk_25",
    "avo_aug_1_chunk_26",
    "avo_aug_1_chunk_27",
    "avo_aug_1_chunk_28",
]

# X
X_full_train = np.load(
    f"{asnwd}/data/plasticc/avocado/avocado_transformed_df_timesteps_100_X_train_avo_aug_1_chunk_0.npy",
)

for file in batch_filename:
    X_temp_train = np.load(
        f"{asnwd}/data/plasticc/avocado/avocado_transformed_df_timesteps_100_X_train_{file}.npy",
    )

    X_full_train = np.concatenate((X_full_train, X_temp_train), axis=0)

np.save(
    f"{asnwd}/data/plasticc/avocado/avocado__transformed_df_timesteps_100_X_full_avo_train.npy",
    X_full_train,
)

# y
y_full_train = np.load(
    f"{asnwd}/data/plasticc/avocado/avocado_transformed_df_timesteps_100_y_train_avo_aug_1_chunk_0.npy",
)

for file in batch_filename:
    y_temp_train = np.load(
        f"{asnwd}/data/plasticc/avocado/avocado_transformed_df_timesteps_100_y_train_{file}.npy",
    )

    y_full_train = np.concatenate((y_full_train, y_temp_train), axis=0)

np.save(
    f"{asnwd}/data/plasticc/avocado/avocado__transformed_df_timesteps_100_y_full_avo_train.npy",
    y_full_train,
)

# Z
Z_full_train = np.load(
    f"{asnwd}/data/plasticc/avocado/avocado_transformed_df_timesteps_100_Z_train_avo_aug_1_chunk_0.npy",
)

for file in batch_filename:
    Z_temp_train = np.load(
        f"{asnwd}/data/plasticc/avocado/avocado_transformed_df_timesteps_100_Z_train_{file}.npy",
    )

    Z_full_train = np.concatenate((Z_full_train, Z_temp_train), axis=0)

np.save(
    f"{asnwd}/data/plasticc/avocado/avocado__transformed_df_timesteps_100_Z_full_avo_train.npy",
    Z_full_train,
)

print(X_full_train.shape, y_full_train.shape, Z_full_train.shape)
