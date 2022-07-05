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
from astronet.utils import load_full_plasticc_test_from_numpy

X_train, y_train, X_test, y_test, Z_train, Z_test = load_full_plasticc_test_from_numpy(
    redshift=True
)

# X_train, y_train, X_test, y_test, loss, Z_train, Z_test = load_dataset(
#     dataset="plasticc",
#     redshift=True,
#     testset=True
# )

X_train_arrays = np.array_split(X_train, 3)
X_train_1 = X_train_arrays[0]
X_train_2 = X_train_arrays[1]
X_train_3 = X_train_arrays[2]
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/X_train_1.npy",
    X_train_1,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/X_train_2.npy",
    X_train_2,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/X_train_3.npy",
    X_train_3,
)

y_train_arrays = np.array_split(y_train, 3)
y_train_1 = y_train_arrays[0]
y_train_2 = y_train_arrays[1]
y_train_3 = y_train_arrays[2]
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/y_train_1.npy",
    y_train_1,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/y_train_2.npy",
    y_train_2,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/y_train_3.npy",
    y_train_3,
)

X_test_arrays = np.array_split(X_test, 3)
X_test_1 = X_test_arrays[0]
X_test_2 = X_test_arrays[1]
X_test_3 = X_test_arrays[2]
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/X_test_1.npy",
    X_test_1,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/X_test_2.npy",
    X_test_2,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/X_test_3.npy",
    X_test_3,
)


y_test_arrays = np.array_split(y_test, 3)
y_test_1 = y_test_arrays[0]
y_test_2 = y_test_arrays[1]
y_test_3 = y_test_arrays[2]
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/y_test_1.npy",
    y_test_1,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/y_test_2.npy",
    y_test_2,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/y_test_3.npy",
    y_test_3,
)

Z_train_arrays = np.array_split(Z_train, 3)
Z_train_1 = Z_train_arrays[0]
Z_train_2 = Z_train_arrays[1]
Z_train_3 = Z_train_arrays[2]
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/Z_train_1.npy",
    Z_train_1,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/Z_train_2.npy",
    Z_train_2,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/Z_train_3k.npy",
    Z_train_3,
)

Z_test_arrays = np.array_split(Z_test, 3)
Z_test_1 = Z_test_arrays[0]
Z_test_2 = Z_test_arrays[1]
Z_test_3 = Z_test_arrays[2]
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/Z_test_1.npy",
    Z_test_1,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/Z_test_2.npy",
    Z_test_2,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/Z_test_3.npy",
    Z_test_3,
)
