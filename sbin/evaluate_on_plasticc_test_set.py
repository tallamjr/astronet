import argparse
import joblib
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shutil
import sys
import seaborn as sns
import tensorflow as tf

from itertools import cycle
from numpy import interp
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from tensorflow import keras

from astronet.constants import astronet_working_directory as asnwd
from astronet.preprocess import one_hot_encode
from astronet.utils import astronet_logger, load_dataset

from astronet.metrics import WeightedLogLoss
from astronet.visualise_results import plot_acc_history, plot_confusion_matrix, plot_loss_history, plot_multiROC

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

import random as python_random
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(RANDOM_SEED)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": ["Computer Modern Roman"]})

print(plt.style.available)

architecture = "t2"
dataset = "plasticc"
X_train, y_train, X_test, y_test, loss, Z_train, Z_test = load_dataset(dataset, redshift=True, augmented=None)
num_classes = y_train.shape[1]

model_name = "1613551066-32f3933"
# model_name = "1613501905-d5b52ec"
# model_name = None

with open(f"{asnwd}/astronet/{architecture}/models/{dataset}/results_with_z.json") as f:
    events = json.load(f)
    if model_name is not None:
    # Get params for model chosen with cli args
        event = next(item for item in events['training_result'] if item["name"] == model_name)
    else:
        # Get params for best model with lowest loss
        event = min(
            (item for item in events["training_result"] if item["augmented"] is None),
                key=lambda ev: ev["model_evaluate_on_test_loss"],
            )

#         event = min(events['training_result'], key=lambda ev: ev['model_evaluate_on_test_loss'])

model = keras.models.load_model(f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{model_name}",
                                custom_objects={'WeightedLogLoss': WeightedLogLoss()},
                               compile=False)

with open(f"{asnwd}/data/full-{dataset}.encoding", "rb") as eb:
    encoding = joblib.load(eb)
class_encoding = encoding.categories_[0]

if dataset == "plasticc":
    class_mapping = {
        90: "SNIa",
        67: "SNIa-91bg",
        52: "SNIax",
        42: "SNII",
        62: "SNIbc",
        95: "SLSN-I",
        15: "TDE",
        64: "KN",
        88: "AGN",
        92: "RRL",
        65: "M-dwarf",
        16: "EB",
        53: "Mira",
        6: "$\mu$-Lens-Single",
    }
    class_encoding
    class_names = list(np.vectorize(class_mapping.get)(class_encoding))
else:

    class_names = class_encoding

from collections import Counter
from pandas.core.common import flatten

y_true = encoding.inverse_transform(y_train)
print(Counter(list(flatten(y_true))))

logloss = event["model_evaluate_on_test_loss"]
acc = event["model_evaluate_on_test_acc"]
print(f"LogLoss on Test Set: {logloss}, Accuracy on Test Set: {acc}")

wloss = WeightedLogLoss()

y_preds = model.predict([X_test, Z_test])
print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")

y_preds_train = model.predict([X_train, Z_train])
print(f"LL-Train: {wloss(y_train, y_preds_train).numpy():.2f}")
# Note the discreptancy seems to be down to inconsistant seeds - see
# https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

cmap = sns.light_palette("Navy", as_cmap=True)
# plot_confusion_matrix(
#     dataset,
#     model_name,
#     y_test,
#     y_preds,
#     encoding,
#     class_names,  # enc.categories_[0]
#     save=True,
#     cmap=cmap
# )

# plot_multiROC(dataset, model_name, model, [X_test, Z_test], y_test, class_names, save=True)

try:
    X_full_test_no_99 = np.load(
        f"{asnwd}/data/plasticc/test_set/full_test_transformed_df_timesteps_100_X_full_test_no_99.npy",
        # mmap_mode='r'
    )

    y_full_test_no_99 = np.load(
        f"{asnwd}/data/plasticc/test_set/full_test_transformed_df_timesteps_100_y_full_test_no_99.npy",
    )

    Z_full_test_no_99 = np.load(
        f"{asnwd}/data/plasticc/test_set/full_test_transformed_df_timesteps_100_Z_full_test_no_99.npy",
    )

except IOError:
    X_full_test = np.load(
        f"{asnwd}/data/plasticc/test_set/full_test_transformed_df_timesteps_100_X_full_test.npy",
        # mmap_mode='r'
    )

    y_full_test = np.load(
        f"{asnwd}/data/plasticc/test_set/full_test_transformed_df_timesteps_100_y_full_test.npy",
    )

    Z_full_test = np.load(
        f"{asnwd}/data/plasticc/test_set/full_test_transformed_df_timesteps_100_Z_full_test.npy",
    )

    print(X_full_test.shape, y_full_test.shape, Z_full_test.shape)

    # Get index of class 99, append index of those NOT 99 to 'keep' list
    class_99_index = []
    for i in range(len(y_full_test.flatten())):
        if (y_full_test.flatten()[i] in [991, 992, 993, 994]):
            pass
        else:
            class_99_index.append(i)

    print(len(class_99_index))

    filter_indices = class_99_index
    axis = 0
    array = X_full_test
    arrayY = y_full_test
    arrayZ = Z_full_test

    X_full_test_no_99 = np.take(array, filter_indices, axis)
    y_full_test_no_99 = np.take(arrayY, filter_indices, axis)
    Z_full_test_no_99 = np.take(arrayZ, filter_indices, axis)

    np.save(
        f"{asnwd}/data/plasticc/test_set/full_test_transformed_df_timesteps_100_X_full_test_no_99.npy",
        X_full_test_no_99,
    )

    np.save(
        f"{asnwd}/data/plasticc/test_set/full_test_transformed_df_timesteps_100_y_full_test_no_99.npy",
        y_full_test_no_99,
    )

    np.save(
        f"{asnwd}/data/plasticc/test_set/full_test_transformed_df_timesteps_100_Z_full_test_no_99.npy",
        Z_full_test_no_99,
    )

print(X_full_test_no_99.shape, y_full_test_no_99.shape, Z_full_test_no_99.shape)

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

enc = enc.fit(y_full_test_no_99)

y_full_test_true_no_99 = enc.transform(y_full_test_no_99)

class_encoding = enc.categories_[0]
if dataset == "plasticc":
    class_mapping = {
        90: "SNIa",
        67: "SNIa-91bg",
        52: "SNIax",
        42: "SNII",
        62: "SNIbc",
        95: "SLSN-I",
        15: "TDE",
        64: "KN",
        88: "AGN",
        92: "RRL",
        65: "M-dwarf",
        16: "EB",
        53: "Mira",
        6: "$\mu$-Lens-Single",
    }
    class_encoding
    class_names = list(np.vectorize(class_mapping.get)(class_encoding))
else:
    class_names = class_encoding

wloss = WeightedLogLoss()

y_preds = model.predict([X_full_test_no_99, Z_full_test_no_99])
print(f"LL-Test: {wloss(y_full_test_true_no_99, y_preds).numpy():.2f}")

cmap = sns.light_palette("Navy", as_cmap=True)
plot_confusion_matrix(
    dataset,
    model_name,
    y_full_test_true_no_99,
    y_preds,
    enc,
    class_names,  # enc.categories_[0]
    save=True,
    cmap=cmap
)

plot_multiROC(dataset, model_name, model, [X_full_test_no_99, Z_full_test_no_99],
        y_full_test_true_no_99, class_names, save=True)
