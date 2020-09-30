from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from astronet.t2.utils import train_val_test_split, create_dataset
from astronet.t2.preprocess import robust_scale, one_hot_encode

import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow import keras

from pathlib import Path
print("File      Path:", Path(__file__).absolute())
print("Parent of Directory Path:", Path().absolute().parent)

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Load WISDM-2010 or WISDM-2019 dataset
column_names = [
    "user_id",
    "activity",
    "timestamp",
    "x_axis",
    "y_axis",
    "z_axis",
]

df = pd.read_csv(str(Path(__file__).absolute().parent.parent.parent) +
    "/data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt",
    header=None,
    names=column_names,
)
df.z_axis.replace(regex=True, inplace=True, to_replace=r";", value=r"")
df["z_axis"] = df.z_axis.astype(np.float64)
df.dropna(axis=0, how="any", inplace=True)

# print(df.head())

cols = ["x_axis", "y_axis", "z_axis"]

# print(df[cols].head())

df_train, df_val, df_test, num_features = train_val_test_split(df, cols)
# print(num_features)  # Should = 3 in this case

# Perfrom robust scaling
robust_scale(df_train, df_val, df_test, cols)

TIME_STEPS = 200
STEP = 40

X_train, y_train = create_dataset(
    df_train[cols],
    df_train.activity,
    TIME_STEPS,
    STEP
)

X_val, y_val = create_dataset(
    df_val[cols],
    df_val.activity,
    TIME_STEPS,
    STEP
)

X_test, y_test = create_dataset(
    df_test[cols],
    df_test.activity,
    TIME_STEPS,
    STEP
)

# print(X_train.shape, y_train.shape)

# One hot encode y
enc, y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test)

with open(str(Path().absolute()) + '/models/results.json') as f:
    events = json.load(f)
    event = max(events['training_result'], key=lambda ev: ev['value'])
    print(event)

model_name = event['name']

model = keras.models.load_model(str(Path().absolute()) + f"/models/model-{model_name}")

model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
cm = confusion_matrix(enc.inverse_transform(y_test), enc.inverse_transform(y_pred))
print(cm / np.sum(cm, axis=1, keepdims=1))

print("             Results for Test Set\n\n" +
      classification_report(enc.inverse_transform(y_pred),
                            enc.inverse_transform(y_test)))
# TODO
# 1. Load saved model.
# 2. Run inference.
# 3. Save plots of performance. import visuals.py plotting functions and call them here.
