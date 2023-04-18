# Copyright 2020 - 2023
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

# Data Processing Pipeline of 'training_alerts'
# Refs: https://zenodo.org/record/7017557#.YyrF-uzMKrM
import pprint
from collections import Counter

import joblib
import numpy as np
import polars as pl
from elasticc.constants import CLASS_MAPPING, ROOT
from sklearn import model_selection
from sklearn.preprocessing import RobustScaler

from astronet.preprocess import one_hot_encode
from astronet.utils import create_dataset

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

cat = "all-classes"

xfeats = (
    False  # Incluce additional features? This will reduce number of possible alerts
)

cat = cat + "-xfeats" if xfeats else cat + "-tsonly"

df = pl.scan_parquet(f"{ROOT}/data/processed/{cat}/class*.parquet").with_columns(
    pl.col("target").cast(pl.Int64).map_dict(CLASS_MAPPING)
)
# (Pdb) df.collect(streaming=True)
# STREAMING CHUNK SIZE: 4545 rows
# shape: (500865200, 11)
# ┌──────────────┬──────────────┬──────────────┬───┬──────────────┬───────────┐
# │ mjd          ┆ lsstu        ┆ lsstg        ┆ … ┆ branch       ┆ uuid      │
# │ ---          ┆ ---          ┆ ---          ┆   ┆ ---          ┆ ---       │
# │ f32          ┆ f32          ┆ f32          ┆   ┆ str          ┆ u32       │
# ╞══════════════╪══════════════╪══════════════╪═══╪══════════════╪═══════════╡
# │ 60295.039062 ┆ 7586.541504  ┆ 18219.744141 ┆ … ┆ Periodic     ┆ 73281911  │
# │ 60297.890625 ┆ 1525.692505  ┆ 1814.988037  ┆ … ┆ Periodic     ┆ 73281911  │
# │ 60300.742188 ┆ 24.350079    ┆ 26.593786    ┆ … ┆ Periodic     ┆ 73281911  │
# │ 60303.589844 ┆ 0.272174     ┆ 0.288707     ┆ … ┆ Periodic     ┆ 73281911  │
# │ …            ┆ …            ┆ …            ┆ … ┆ …            ┆ …         │
# │ 60605.039062 ┆ -3662.459229 ┆ -3712.712402 ┆ … ┆ Non-Periodic ┆ 155242066 │
# │ 60607.058594 ┆ -3648.506836 ┆ -3636.555176 ┆ … ┆ Non-Periodic ┆ 155242066 │
# │ 60609.074219 ┆ -3610.797363 ┆ -3530.018555 ┆ … ┆ Non-Periodic ┆ 155242066 │
# │ 60611.089844 ┆ -3541.750977 ┆ -3402.252686 ┆ … ┆ Non-Periodic ┆ 155242066 │
# └──────────────┴──────────────┴──────────────┴───┴──────────────┴───────────┘
test_df = (
    df.select(["object_id", "target", "branch", "uuid"])
    .groupby("object_id")
    .agg([pl.count(), pl.col("target"), pl.col("branch")])
    .filter(pl.col("branch").arr.unique().arr.lengths() > 1)
    .filter(pl.col("target").arr.unique().arr.lengths() > 1)
    .collect(streaming=True)
)
# Ensure each object_id is associated to one branch and target value.
assert test_df.height == 0
print(df.head().collect())
df = df.collect(streaming=True)
print(df.height)

num_filters = 6
num_gps = 100

Xs = df.select("^lsst.*$").to_numpy().reshape((df.height // 100), 100, 6)

tab = df.select(["object_id", "target", "uuid", "branch"]).unique(
    subset="object_id", maintain_order=True
)

ys = tab.select("target").to_numpy()
bs = tab.select("branch").to_numpy()
groups = tab.select("uuid").to_numpy().flatten()

print(groups.shape)

# gss = model_selection.StratifiedGroupKFold(n_splits=2, random_state=RANDOM_SEED)

gss = model_selection.GroupShuffleSplit(
    n_splits=1, random_state=RANDOM_SEED, test_size=None, train_size=0.7
)

gss.get_n_splits()

print(gss)

for i, (train_index, test_index) in enumerate(gss.split(Xs, ys, groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")

np.save(f"{ROOT}/data/processed/{cat}/groups.npy", groups)
np.save(f"{ROOT}/data/processed/{cat}/groups_train_idx.npy", groups[train_index])

X_train = Xs[train_index]
X_test = Xs[test_index]

y_train = ys[train_index]
y_test = ys[test_index]

scaler = RobustScaler()
X_train = X_train.reshape(X_train.shape[0] * num_gps, num_filters)
X_train = scaler.fit(X_train).transform(X_train)
X_train = X_train.reshape(X_train.shape[0] // num_gps, num_gps, num_filters)

scaler = RobustScaler()
X_test = X_test.reshape(X_test.shape[0] * num_gps, num_filters)
X_test = scaler.fit(X_test).transform(X_test)
X_test = X_test.reshape(X_test.shape[0] // num_gps, num_gps, num_filters)

print("\nTRAIN COUNTER:\n")
pprint.pprint(Counter(y_train.squeeze()))
print("\nTEST COUNTER:\n")
pprint.pprint(Counter(y_test.squeeze()))
# check same classes in train appear in test
assert set(np.unique(y_train)) == set(np.unique(y_test))

# One hot encode y, all classes
enc, y_train, y_test = one_hot_encode(y_train, y_test)
encoding_file = f"{ROOT}/data/processed/{cat}/all-classes.enc"

with open(encoding_file, "wb") as f:
    joblib.dump(enc, f)

print("SAVING NEW DATASET")

# passbands
np.save(f"{ROOT}/data/processed/{cat}/X_train.npy", X_train)
np.save(f"{ROOT}/data/processed/{cat}/X_test.npy", X_test)

# labels
np.save(f"{ROOT}/data/processed/{cat}/y_train.npy", y_train)
np.save(f"{ROOT}/data/processed/{cat}/y_test.npy", y_test)

# branches
y_train_bs = bs[train_index]
y_test_bs = bs[test_index]

print("\nTRAIN COUNTER:\n")
pprint.pprint(Counter(y_train_bs.squeeze()))
print("\nTEST COUNTER:\n")
pprint.pprint(Counter(y_test_bs.squeeze()))
# check same classes in train appear in test
assert set(np.unique(y_train_bs)) == set(np.unique(y_test_bs))

# One hot encode y, sub classes (branches)
enc, y_train_bs, y_test_bs = one_hot_encode(y_train_bs, y_test_bs)
encoding_file = f"{ROOT}/data/processed/{cat}/branches.enc"

with open(encoding_file, "wb") as f:
    joblib.dump(enc, f)

np.save(f"{ROOT}/data/processed/{cat}/y_train_bs.npy", y_train_bs)
np.save(f"{ROOT}/data/processed/{cat}/y_test_bs.npy", y_test_bs)

if xfeats:
    z = ["z", "z_error"]

    # redshift
    Zs, ys, _ = create_dataset(
        df.select(["z", "z_error"]),
        df.select("target"),
        df.select("uuid"),
        time_steps=num_gps,
        step=100,
    )

    Z_train = Zs[train_index]
    Z_test = Zs[test_index]

    Z_train = np.mean(Z_train, axis=1)
    Z_test = np.mean(Z_test, axis=1)

    scaler = RobustScaler()
    Z_train = scaler.fit(Z_train).transform(Z_train)

    scaler = RobustScaler()
    Z_test = scaler.fit(Z_test).transform(Z_test)

    # other feats
    zplus = [
        "ra",
        "dec",
        "hostgal_ra",
        "hostgal_dec",
        "nobs",
    ]

    if zplus:
        Zs, ys, _ = create_dataset(
            df.select(zplus),
            df.select("target"),
            df.select("uuid"),
            time_steps=num_gps,
            step=100,
        )

        Z_train_add = Zs[train_index]
        Z_test_add = Zs[test_index]

        Z_train_add = np.mean(Z_train_add, axis=1)
        Z_test_add = np.mean(Z_test_add, axis=1)

        Z_train = np.hstack((Z_train, Z_train_add))
        Z_test = np.hstack((Z_test, Z_test_add))

        z.extend(zplus)

    xfeatures = "_".join(z)

    # additional features
    np.save(f"{ROOT}/data/processed/{cat}/Z_train_{xfeatures}.npy", Z_train)
    np.save(f"{ROOT}/data/processed/{cat}/Z_test_{xfeatures}.npy", Z_test)

    print(
        f"TRAIN SHAPES:\n x = {X_train.shape} \n z = {Z_train.shape} \n y = {y_train.shape}"
    )
    print(
        f"TEST SHAPES:\n x = {X_test.shape} \n z = {Z_test.shape} \n y = {y_test.shape} \n"
    )
else:
    print(
        f"TRAIN SHAPES:\n x = {X_train.shape} \n y = {y_train.shape} \n b = {y_train_bs.shape}"
    )
    print(
        f"TEST SHAPES:\n x = {X_test.shape} \n y = {y_test.shape} \n b = {y_test_bs.shape}"
    )
