import argparse
import json
import os
import shutil
import sys
from itertools import cycle
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from numpy import interp
from sklearn.metrics import auc, confusion_matrix, roc_curve
from tensorflow import keras

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.preprocess import one_hot_encode
from astronet.utils import astronet_logger, load_dataset


def update_results(mode: str = "precision"):

    if mode not in ["precision", "recall"]:
        results_key = f"model_predict_{mode}_score"
    else:
        results_key = "model_evaludate_on_test_acc"

    table = {}
    for dataset in datasets:
        print(f"{dataset}")

        X_train, y_train, X_test, y_test, loss = load_dataset(dataset)

        with open(
            f"{asnwd}/astronet/{architecture}/models/{dataset}/results.json"
        ) as f:
            events = json.load(f)

        # Get params for best model with lowest precision score
        event = max(events["training_result"], key=lambda ev: ev[results_key])

        model_name = event["name"]

        table[f"{dataset}"] = event[results_key]

    print(table)

    df = pd.DataFrame.from_dict(table, orient="index")
    df.columns = [f"{architecture}"]

    filename = f"{asnwd}/results/mts-{architecture}-results-{mode}.csv"
    df.to_csv(filename)

    return df


def update_accuracy_results():
    pass


def update_recall_results():
    pass
