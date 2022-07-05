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

import argparse
import json
import os
import random as python_random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
from pandas.core.common import flatten
from tensorflow import keras

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.constants import LOCAL_DEBUG
from astronet.metrics import WeightedLogLoss
from astronet.utils import (
    astronet_logger,
    find_optimal_batch_size,
    get_encoding,
)
from astronet.viz.visualise_results import (
    plot_acc_history,
    plot_confusion_matrix,
    plot_confusion_matrix_against_baseline,
    plot_loss_history,
    plot_multiPR,
    plot_multiROC,
)

try:
    log = astronet_logger(__file__)
    log.info("\n" + "=" * (shutil.get_terminal_size((80, 20))[0]))
    log.info(f"File Path: {Path(__file__).absolute()}")
    log.info(f"Parent of Directory Path: {Path().absolute().parent}")
except Exception as e:
    print(f"{e}: Seems you are running from a notebook...")
    __file__ = f"{Path().resolve().parent}/astronet/viz/generate_plots.py"
    log = astronet_logger(__file__)

np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.2f}".format})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(RANDOM_SEED)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.serif": ["Computer Modern Roman"],
    }
)


class Plots(object):
    # TODO: Update docstrings
    def __init__(self, architecture, dataset, model_name, redshift, ztf, savefigs=True):
        self.architecture = architecture
        self.dataset = dataset
        self.model_name = model_name
        self.redshift = redshift
        self.savefigs = savefigs
        self.ztf = ztf

    def __call__(self):

        start = time.time()
        X_test = np.load(
            f"{asnwd}/data/plasticc/test_set/infer/X_test.npy",
        )
        y_test = np.load(
            f"{asnwd}/data/plasticc/test_set/infer/y_test.npy",
        )
        Z_test = np.load(
            f"{asnwd}/data/plasticc/test_set/infer/Z_test.npy",
        )

        X_test = X_test[:, :, 0:3:2] if self.ztf is not None else X_test
        print(f"X_TEST: {X_test.shape}, Y_TEST: {y_test.shape}, Z_TEST: {Z_test.shape}")

        (
            num_samples,
            timesteps,
            num_features,
        ) = X_test.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)

        BATCH_SIZE = find_optimal_batch_size(num_samples)
        print(f"BATCH_SIZE:{BATCH_SIZE}")

        if self.redshift is not None:
            test_input = [X_test, Z_test]
            test_ds = (
                tf.data.Dataset.from_tensor_slices(
                    ({"input_1": test_input[0], "input_2": test_input[1]}, y_test)
                )
                .batch(BATCH_SIZE, drop_remainder=False)
                .prefetch(tf.data.AUTOTUNE)
            )

            results_filename = f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/results_with_z.json"

        else:
            test_input = X_test
            test_ds = (
                tf.data.Dataset.from_tensor_slices((test_input, y_test))
                .batch(BATCH_SIZE, drop_remainder=False)
                .prefetch(tf.data.AUTOTUNE)
            )

            results_filename = f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/results.json"

        with open(results_filename) as f:
            events = json.load(f)
            if self.model_name is not None:
                # Get params for model chosen with cli args
                event = next(
                    item
                    for item in events["training_result"]
                    if item["name"] == self.model_name
                )
            else:
                event = min(
                    events["training_result"],
                    key=lambda ev: ev["model_evaluate_on_test_loss"],
                )

        model = keras.models.load_model(
            f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/model-{self.model_name}",
            custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            compile=False,
        )

        dataform = "testset"
        encoding, class_encoding, class_names = get_encoding(
            self.dataset, dataform=dataform
        )

        y_true_test = encoding.inverse_transform(y_test)
        print("N_TEST:", Counter(list(flatten(y_true_test))))

        logloss = event["model_evaluate_on_test_loss"]
        acc = event["model_evaluate_on_test_acc"]
        print(f"LogLoss on Test Set: {logloss}, Accuracy on Test Set: {acc}")

        y_test_ds = (
            tf.data.Dataset.from_tensor_slices(y_test)
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        print("Running predictions")
        wloss = WeightedLogLoss()

        if LOCAL_DEBUG is not None:
            log.info("LOCAL_DEBUG set, reducing dataset size...")
            test_ds = test_ds.take(300)
            y_test_ds = y_test_ds.take(300)

        # initialize tqdm callback with default parameters
        tqdm_callback = tfa.callbacks.TQDMProgressBar()
        y_preds = model.predict(test_ds, callbacks=[tqdm_callback], verbose=2)
        y_test_np = np.concatenate([y for y in y_test_ds], axis=0)

        loss = wloss(y_test_np, y_preds).numpy()
        print(f"LL-Test: {loss:.3f}")

        # Causes OOM with all samples -->
        # y_preds = model.predict([X_test, Z_test], batch_size=num_samples)
        # print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")

        print("Plotting figures...")
        cmap = sns.light_palette("Navy", as_cmap=True)
        plot_confusion_matrix(
            self.architecture,
            self.dataset,
            self.model_name,
            y_test_np,
            y_preds,
            encoding,
            class_names,  # enc.categories_[0]
            save=self.savefigs,
            cmap=cmap,
        )
        log.info("CM DONE...")

        # cmap = sns.light_palette("purple", as_cmap=True)
        # plot_confusion_matrix_against_baseline(
        #     self.architecture,
        #     self.dataset,
        #     self.model_name,
        #     test_ds,
        #     y_test_np,
        #     y_preds,
        #     encoding,
        #     class_names,  # enc.categories_[0]
        #     save=self.savefigs,
        #     cmap=cmap,
        # )
        # log.info("CMB DONE...")

        plot_acc_history(
            self.architecture, self.dataset, self.model_name, event, save=self.savefigs
        )
        log.info("ACC DONE...")

        plot_loss_history(
            self.architecture, self.dataset, self.model_name, event, save=self.savefigs
        )
        log.info("LOSS DONE...")

        plot_multiROC(
            self.architecture,
            self.dataset,
            self.model_name,
            model,
            y_test_np,
            y_preds,
            class_names,
            save=self.savefigs,
        )
        log.info("ROC DONE...")

        plot_multiPR(
            self.architecture,
            self.dataset,
            self.model_name,
            model,
            y_test_np,
            y_preds,
            class_names,
            save=self.savefigs,
        )
        log.info("PR DONE...")

        end = time.time()
        log.info(f"PLOTS GENERATED IN {(end - start) / 60:.2f} MINUTES")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate trained model for given architecture"
    )

    parser.add_argument(
        "-a",
        "--architecture",
        default="t2",
        help="Choose which architecture to evaluate : 'atx', 't2'",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        default="plasticc",
        help="Choose which dataset to use: This is fixed for plasticc for now",
    )

    parser.add_argument(
        "-m",
        "--model",
        default="1619624444-0.1.dev765+g7c90cbb.d20210428",
        help="Name of tensorflow.keras model, i.e. model-<timestamp>-<hash>",
    )

    parser.add_argument(
        "-z",
        "--redshift",
        default=None,
        help="Whether to include redshift features or not",
    )

    parser.add_argument(
        "-f",
        "--ztf",
        default=None,
        help="Model trained on ZTF-esque data or not",
    )

    try:
        args = parser.parse_args()
        argsdict = vars(args)
    except KeyError:
        parser.print_help()
        sys.exit(0)

    architecture = args.architecture
    dataset = args.dataset
    model_name = args.model
    redshift = args.redshift
    ztf = args.ztf

    if redshift is not None:
        redshift = True

    if ztf is not None:
        ztf = True

    plotting = Plots(
        architecture=architecture,
        dataset=dataset,
        model_name=model_name,
        redshift=redshift,
        ztf=ztf,
    )
    plotting()
